import os
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import gc
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.distributions import Categorical

from pyscipopt import Model, quicksum, SCIP_RESULT, Heur, Eventhdlr, SCIP_EVENTTYPE, SCIP_PARAMSETTING, SCIP_HEURTIMING

from hardware_setup import device, MACHINE_SPEED_FACTOR
from optical_physics import OpticalPhysicsCalculator, QoTPreprocessor, generate_topology, generate_dynamic_demands, build_qot_rsa_ilp
from matheuristics import RSAMatheuristics

# ==========================================
# 3. RL ENVIRONMENT, LPE/SPE EXTRACTOR & GNN
# ==========================================
class PrimalIntegralTracker(Eventhdlr):
    def __init__(self, speed_factor=1.0):
        super().__init__()
        self.speed_factor = speed_factor
        self.time_stamps, self.gap_values, self.start_wall_time = [0.0], [1.0], None
        
    def eventinit(self): self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
    def eventexit(self): self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
    def init_solve(self): self.start_wall_time = time.perf_counter()
        
    def eventexec(self, event):
        if self.start_wall_time is None: return {}
        try:
            calibrated_now = (time.perf_counter() - self.start_wall_time) / self.speed_factor
            pb, db = self.model.getPrimalbound(), self.model.getDualbound()
            gap = 0.0 if abs(db) < 1e-6 and abs(pb) < 1e-6 else (1.0 if pb >= 1e19 else min(1.0, abs(pb - db) / max(abs(pb), abs(db))))
            self.time_stamps.append(calibrated_now)
            self.gap_values.append(gap)
        except: pass
        return {}
    
    def get_integral(self, t_max):
        integral = sum(self.gap_values[i-1] * (self.time_stamps[i] - self.time_stamps[i-1]) for i in range(1, len(self.time_stamps)))
        calibrated_t_max = t_max / self.speed_factor
        return integral + (self.gap_values[-1] * (calibrated_t_max - self.time_stamps[-1])) if self.time_stamps[-1] < calibrated_t_max else integral

class RSAData:
    def __init__(self, x_v, x_c, edge_index, link_indices, slot_indices):
        self.x_v, self.x_c, self.edge_index = x_v.to(device), x_c.to(device), edge_index.to(device)
        self.link_indices = torch.tensor(link_indices, dtype=torch.long, device=device)
        self.slot_indices = torch.tensor(slot_indices, dtype=torch.long, device=device)
        self.x_o = torch.mean(self.x_v, dim=0, keepdim=True) if self.x_v.size(0) > 0 else torch.zeros((1, 17), device=device)

class LPE_SPE_StateExtractor:
    def __init__(self, network, total_slots):
        self.network = network
        self.total_slots = total_slots
        self.arcs = list(network.edges()) + [(v, u) for u, v in network.edges()]
        self.arc_to_idx = {arc: i for i, arc in enumerate(self.arcs)}
        self.num_links = len(self.arcs)

    def extract_state(self, model):
        vars_scip = model.getVars(transformed=True)
        conss_scip = model.getConss(transformed=True)
        lp_opt = model.getLPSolstat() == "optimal"
        
        x_v_np = np.zeros((len(vars_scip), 17), dtype=np.float32)
        link_indices = []
        slot_indices = []
        var_map = {}
        
        for i, var in enumerate(vars_scip):
            var_map[var.name] = i
            sol = model.getSolVal(None, var) if lp_opt else 0.0
            
            x_v_np[i, :] = [var.getObj(), 1.0 if var.vtype() == "BINARY" else 0.0, 0.0, 0.0,
                            var.getLbLocal(), var.getUbLocal(), 1.0 if sol == var.getLbLocal() else 0.0,
                            1.0 if sol == var.getUbLocal() else 0.0, sol, sol % 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            l_idx, s_idx = 0, 0 
            parts = var.name.split('_')
            if len(parts) == 5 and parts[0] == 'f':
                try:
                    u, v, s = int(parts[2]), int(parts[3]), int(parts[4])
                    if (u, v) in self.arc_to_idx: l_idx = self.arc_to_idx[(u, v)]
                    s_idx = s
                except: pass
                
            link_indices.append(l_idx)
            slot_indices.append(s_idx)
                            
        x_c_np = np.zeros((len(conss_scip), 4), dtype=np.float32)
        edge_rows, edge_cols = [], []
        
        for i, cons in enumerate(conss_scip):
            try:
                for var_name in model.getValsLinear(cons).keys():
                    if var_name in var_map:
                        edge_rows.append(i)
                        edge_cols.append(var_map[var_name])
            except: pass
            
        return RSAData(torch.from_numpy(x_v_np), torch.from_numpy(x_c_np), 
                       torch.tensor([edge_rows, edge_cols], dtype=torch.long),
                       link_indices, slot_indices)

class BipartiteGCN_RSA(nn.Module):
    def __init__(self, num_links, num_slots, emb_dim=64, num_actions=5):
        super().__init__()
        self.v_emb = nn.Linear(17, emb_dim)
        self.c_emb = nn.Linear(4, emb_dim)
        self.lpe_embedding = nn.Embedding(num_links + 1, emb_dim)
        self.spe_embedding = nn.Embedding(num_slots + 1, emb_dim)
        self.conv_v2c = SAGEConv((emb_dim, emb_dim), emb_dim)
        self.conv_c2v = SAGEConv((emb_dim, emb_dim), emb_dim)
        self.policy_head = nn.Sequential(nn.Linear(emb_dim, 32), nn.ReLU(), nn.Linear(32, num_actions), nn.Softmax(dim=-1))
        self.value_head = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())

    def forward(self, data):
        hv = F.relu(self.v_emb(data.x_v))
        hc = F.relu(self.c_emb(data.x_c))
        if hv.size(0) > 0: hv = hv + self.lpe_embedding(data.link_indices) + self.spe_embedding(data.slot_indices)
        if hv.size(0) > 0 and hc.size(0) > 0 and data.edge_index.size(1) > 0:
            hc = F.relu(self.conv_v2c((hv, hc), torch.stack([data.edge_index[1], data.edge_index[0]])))
            hv = F.relu(self.conv_c2v((hc, hv), data.edge_index))
        global_state = torch.mean(hv, dim=0) if hv.size(0) > 0 else torch.zeros(self.v_emb.out_features, device=hv.device)
        return self.policy_head(global_state), self.value_head(hv)

class TripartiteGCN_RSA(nn.Module):
    def __init__(self, num_links, num_slots, emb_dim=64, num_actions=5):
        super().__init__()
        self.v_emb = nn.Linear(17, emb_dim)
        self.c_emb = nn.Linear(4, emb_dim)
        self.o_emb = nn.Linear(17, emb_dim) 
        self.lpe_embedding = nn.Embedding(num_links + 1, emb_dim)
        self.spe_embedding = nn.Embedding(num_slots + 1, emb_dim)
        self.conv_v2c = SAGEConv((emb_dim, emb_dim), emb_dim)
        self.conv_c2v = SAGEConv((emb_dim, emb_dim), emb_dim)
        self.policy_head = nn.Sequential(nn.Linear(emb_dim * 2, 32), nn.ReLU(), nn.Linear(32, num_actions), nn.Softmax(dim=-1))
        self.value_head = nn.Sequential(nn.Linear(emb_dim, 1), nn.Sigmoid())

    def forward(self, data):
        hv = F.relu(self.v_emb(data.x_v))
        hc = F.relu(self.c_emb(data.x_c))
        ho = F.relu(self.o_emb(data.x_o)) 
        if hv.size(0) > 0: hv = hv + self.lpe_embedding(data.link_indices) + self.spe_embedding(data.slot_indices)
        if hv.size(0) > 0 and hc.size(0) > 0 and data.edge_index.size(1) > 0:
            hc = F.relu(self.conv_v2c((hv, hc), torch.stack([data.edge_index[1], data.edge_index[0]])))
            hv = F.relu(self.conv_c2v((hc, hv), data.edge_index))
        global_state = torch.mean(hv, dim=0) if hv.size(0) > 0 else torch.zeros(self.v_emb.out_features, device=hv.device)
        tripartite_state = torch.cat([global_state.unsqueeze(0), ho], dim=-1).squeeze(0)
        return self.policy_head(tripartite_state), self.value_head(hv)

class RSAGNN_Driver(nn.Module):
    def __init__(self, architecture, num_links, num_slots, num_actions=5, emb_dim=64):
        super().__init__()
        if architecture == "tripartite":
            self.model = TripartiteGCN_RSA(num_links=num_links, num_slots=num_slots, emb_dim=emb_dim, num_actions=num_actions)
        else:
            self.model = BipartiteGCN_RSA(num_links=num_links, num_slots=num_slots, emb_dim=emb_dim, num_actions=num_actions)

    def forward(self, rsa_data):
        return self.model(rsa_data)

# ==========================================
# 4. THE SUCCESS ORACLE
# ==========================================
class RLHeuristicScheduler(Heur):
    def __init__(self, gnn_driver, state_extractor, muscle, actual_model, stealth_mode=False):
        super().__init__()
        self.gnn_driver = gnn_driver
        self.state_extractor = state_extractor
        self.muscle = muscle
        self.actual_model = actual_model 
        self.stealth_mode = stealth_mode
        self.saved_log_probs, self.saved_actions, self.saved_entropies = [], [], []
        
        # Max of 5 interventions so the agent acts globally at the top of the tree 
        self.intervention_count = 0
        self.max_interventions = 5 
        
    def heurexec(self, heistage, heumnode):
        if self.intervention_count >= self.max_interventions:
            return {"result": SCIP_RESULT.DIDNOTRUN}
            
        rsa_data = self.state_extractor.extract_state(self.actual_model)
        if rsa_data.x_v.size(0) == 0: return {"result": SCIP_RESULT.DIDNOTRUN}
            
        action_probs, var_assignments = self.gnn_driver(rsa_data)
        m = Categorical(action_probs)
        action = m.sample()
        
        self.saved_log_probs.append(m.log_prob(action))
        self.saved_actions.append(action.item() + 1)
        self.saved_entropies.append(m.entropy()) 
        
        self.intervention_count += 1
        
        var_names = [v.name for v in self.actual_model.getVars(transformed=True)]
        var_prob_dict = {name: prob.item() for name, prob in zip(var_names, var_assignments.flatten())} if var_assignments is not None else {}

        new_sol_dict = None
        action_idx = action.item() + 1 # Action Space matches indices 1-5 from thesis
        
        if action_idx == 1: new_sol_dict = self.muscle.run_neural_diving(self.actual_model, var_prob_dict, top_k_percent=0.05)
        elif action_idx == 2: new_sol_dict = self.muscle.run_local_branching(self.actual_model, k_size=5)
        elif action_idx == 3: new_sol_dict = self.muscle.execute_decomposition(self.actual_model, strategy='SA-R', use_absm=False)
        elif action_idx == 4: new_sol_dict = self.muscle.run_tabu_search(self.actual_model, use_absm=False)
        elif action_idx == 5: new_sol_dict = self.muscle.run_hai_genetic_algorithm(self.actual_model, use_absm=False)

        if new_sol_dict is not None:
            if self.stealth_mode:
                return {"result": SCIP_RESULT.DIDNOTFIND}
                
            sol = self.actual_model.createSol(self)
            main_vars = {v.name: v for v in self.actual_model.getVars(transformed=True)}
            
            for var_name, val in new_sol_dict.items():
                if var_name in main_vars:
                    self.actual_model.setSolVal(sol, main_vars[var_name], val)
                    
            accepted = self.actual_model.trySol(sol)
            if accepted: 
                return {"result": SCIP_RESULT.FOUNDSOL}
                
        return {"result": SCIP_RESULT.DIDNOTFIND}

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def solve_instance(G_undirected, demands, num_slots, forbidden_arcs, t_max, driver=None, extractor=None, muscle=None):
    model, _ = build_qot_rsa_ilp(G_undirected, demands, num_slots, forbidden_arcs)
    model.setRealParam("limits/time", t_max)
    model.setLongintParam("limits/nodes", 300) 
    model.setIntParam("limits/maxsol", 5)
    model.hideOutput()
    
    tracker = PrimalIntegralTracker(speed_factor=MACHINE_SPEED_FACTOR)
    model.data = {'tracker': tracker} 
    model.includeEventhdlr(tracker, "Tracker", "Tracks PIO")
    
    scheduler = None
    if driver is not None:
        model.setHeuristics(SCIP_PARAMSETTING.FAST) 
        scheduler = RLHeuristicScheduler(driver, extractor, muscle, model)
        model.includeHeur(scheduler, "RL_Scheduler", "Success Oracle", "Y", timingmask=SCIP_HEURTIMING.BEFORENODE)
        model.setParam("heuristics/RL_Scheduler/freq", 15) 
        
    tracker.init_solve()
    model.optimize()
    
    pio = tracker.get_integral(t_max)
    
    try:
        model.freeTransform()
    except: pass
    
    tracker.model = None 
    model.data = {} 
    
    return pio, scheduler, model 

def train_success_oracle(architecture="tripartite", topologies=["SPAIN", "NSFNET", "COST239"], episodes=500, save_dir="training_data"):
    os.makedirs(save_dir, exist_ok=True)
    num_slots, t_max = 60, 180.0 
    
    max_edges = max([len(generate_topology(t).edges()) for t in topologies]) * 2
    
    driver = RSAGNN_Driver(architecture=architecture, num_links=max_edges, num_slots=num_slots, emb_dim=64, num_actions=5)
    driver.to(device)
    # Higher Learning Rate for stable convergence
    optimizer = torch.optim.Adam(driver.parameters(), lr=1e-3)
    
    history = {"episode": [], "agent_pio": [], "default_pio": [], "loss": []}
    
    for ep in range(episodes):
        # --------------------------------------------------------- 
        # OVERFIT MECHANISM: 30 FIXED SCENARIOS
        # ---------------------------------------------------------
        scenario_id = ep % 30
        random.seed(42 + scenario_id)
        np.random.seed(42 + scenario_id)
        
        topo_name = topologies[scenario_id % len(topologies)]
        G_undirected = generate_topology(topo_name)
        num_reqs = random.randint(4, 6)
        demands = generate_dynamic_demands(G_undirected, n_requests=num_reqs)
        
        random.seed()
        np.random.seed()
        # ---------------------------------------------------------

        preprocessor = QoTPreprocessor(G_undirected, demands)
        forbidden_arcs = preprocessor.combined_preprocessing()
        
        print(f"Ep {ep+1}/{episodes} | Inst: {topo_name}_{num_reqs} | Running Default SCIP...")
        default_pio, _, default_model = solve_instance(G_undirected, demands, num_slots, forbidden_arcs, t_max)
        del default_model 
        
        print(f"Ep {ep+1}/{episodes} | Inst: {topo_name}_{num_reqs} | Running Agent SCIP...")
        muscle = RSAMatheuristics(demands, G_undirected, num_slots, forbidden_arcs)
        extractor = LPE_SPE_StateExtractor(G_undirected, num_slots)
        agent_pio, scheduler, agent_model = solve_instance(G_undirected, demands, num_slots, forbidden_arcs, t_max, driver, extractor, muscle)
        
        advantage = agent_pio - default_pio 
        
        # SCIENTIFIC FIX: Scale advantage to prevent gradient whiplash
        scaled_advantage = advantage / (default_pio + 1e-5)
            
        loss_val = 0.0
        if scheduler and scheduler.saved_log_probs:
            policy_loss = (torch.stack(scheduler.saved_log_probs).sum() * scaled_advantage)
            entropy_bonus = torch.stack(scheduler.saved_entropies).mean()
            loss = policy_loss - (0.01 * entropy_bonus)

            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(driver.parameters(), 1.0)
                optimizer.step()
                loss_val = loss.item()
                
        print(f"   => Default PIO: {default_pio:.2f} | Agent PIO: {agent_pio:.2f} | Advantage: {advantage:.2f}")
        history["episode"].append(ep + 1)
        history["agent_pio"].append(agent_pio)
        history["default_pio"].append(default_pio)
        history["loss"].append(loss_val)
        
        if scheduler:
            scheduler.actual_model = None 
            scheduler.gnn_driver = None
            del scheduler.saved_log_probs[:]
            del scheduler.saved_actions[:]
            del scheduler.saved_entropies[:]
            del scheduler
            
        del agent_model
        del muscle, extractor, demands, forbidden_arcs, G_undirected
        
        gc.collect()
        torch.cuda.empty_cache() 
        
        if ep > 0 and ep % 50 == 0: torch.save(driver.state_dict(), os.path.join(save_dir, f"{architecture}_weights-{ep}.pth"))
        
    torch.save(driver.state_dict(), os.path.join(save_dir, f"{architecture}_weights.pth"))
    return pd.DataFrame(history), driver

# ==========================================
# 6. BENCHMARKING & REPORTING
# ==========================================
def calculate_fragmentation_index(usage_matrix):
    frag_indices = []
    for arc_usage in usage_matrix:
        used_slots = np.where(arc_usage > 0.5)[0]
        if len(used_slots) == 0: continue
        max_slot, total_used = used_slots[-1], len(used_slots)
        if max_slot > 0: frag_indices.append((max_slot + 1 - total_used) / (max_slot + 1))
    return np.mean(frag_indices) if frag_indices else 0.0

def _build_usage_from_sol(model, sol, G, demands, total_slots):
    usage = np.zeros((len(G.edges()) * 2, total_slots))
    arcs = list(G.edges())
    arc_to_idx = {arc: i for i, arc in enumerate(arcs + [(v, u) for u, v in arcs])}
    demands_dict = {d['id']: d for d in demands}
    for var in model.getVars():
        val = model.getSolVal(sol, var)
        if val > 0.5:
            parts = var.name.split('_')
            if len(parts) == 5 and parts[0] == "f":
                k_id, u, v, s = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                arc_idx = arc_to_idx.get((u, v))
                if arc_idx is not None:
                    for slot in range(s - demands_dict[k_id]['slots'] + 1, s + 1):
                        if 0 <= slot < total_slots: usage[arc_idx, slot] = 1.0
    return usage

def run_exhaustive_benchmarks(trained_driver, num_slots, topologies, architecture):
    loads = [5, 10]
    baselines = ["Default SCIP", "Neural Scheduler"]
    results, convergence_data, action_logs = [], {}, []
    
    for topo in topologies:
        print(f"\n[Benchmark] Topology: {topo}")
        network_G = generate_topology(topo)
        for load in loads:
            print(f"  -> Load: {load} Demands")
            demands = generate_dynamic_demands(network_G, n_requests=load)
            preprocessor = QoTPreprocessor(network_G, demands)
            forbidden_arcs = preprocessor.combined_preprocessing()

            for baseline in baselines:
                print(f"    * Running {baseline}...")
                model, _ = build_qot_rsa_ilp(network_G, demands, num_slots, forbidden_arcs)
                model.setRealParam("limits/time", 360.0) 
                model.hideOutput()
                
                tracker = PrimalIntegralTracker(speed_factor=MACHINE_SPEED_FACTOR)
                model.data = {'tracker': tracker} 
                model.includeEventhdlr(tracker, "Tracker", "Tracks PIO")
                
                muscle = RSAMatheuristics(demands, network_G, num_slots, forbidden_arcs)
                
                if baseline == "Neural Scheduler":
                    model.setHeuristics(SCIP_PARAMSETTING.FAST) 
                    extractor = LPE_SPE_StateExtractor(network_G, num_slots)
                    scheduler = RLHeuristicScheduler(trained_driver, extractor, muscle, model)
                    model.includeHeur(scheduler, "RL_Scheduler", "Oracle", "Y", timingmask=SCIP_HEURTIMING.BEFORENODE)
                    model.setParam("heuristics/RL_Scheduler/freq", 15)
                
                tracker.init_solve()
                model.optimize()
                
                pio = tracker.get_integral(360.0)
                best_sol = model.getBestSol()
                
                if best_sol:
                    # DATA SANITIZATION: Cast internal SCIP floats to strict Python integers
                    max_slot = int(round(model.getSolObjVal(best_sol)))
                    usage_matrix = _build_usage_from_sol(model, best_sol, network_G, demands, num_slots)
                    frag_index = calculate_fragmentation_index(usage_matrix)
                else:
                    max_slot, frag_index = float('inf'), 1.0
                
                results.append({"Topology": topo, "Load": load, "Baseline": baseline, "PIO": pio, 
                                "Gap %": model.getGap() * 100, "Max Slot": max_slot, "Fragmentation": frag_index})
                
                if load == loads[-1]: convergence_data[f"{topo}_{baseline}"] = (tracker.time_stamps, tracker.gap_values)
                if baseline == "Neural Scheduler" and hasattr(scheduler, 'saved_actions'): action_logs.extend(scheduler.saved_actions)

    return pd.DataFrame(results), convergence_data, pd.Series(action_logs).value_counts()</content>
<parameter name="filePath">d:\Shankar\goat-ipython\goat-vault\goat-vault\01 - Notes\03 - Resources\France\acads\Research Project\publish-code\rl_components.py