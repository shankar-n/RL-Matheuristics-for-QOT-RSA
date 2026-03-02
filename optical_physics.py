import random
import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

# ==========================================
# 1. OPTICAL PHYSICS & PREPROCESSING
# ==========================================
class OpticalPhysicsCalculator:
    def __init__(self):
        self.alpha_dB = 0.2
        self.alpha = self.alpha_dB / (10 * np.log10(np.exp(1))) 
        self.d_coeff = 17.0  
        self.gamma = 1.3e-3  
        self.beta2 = 21.667e-27 
        self.NF = 5.0 
        self.n_sp = (10 ** (self.NF / 10)) / 2
        self.B_WDM = 5000e9 
        self.B_N = 12.5e9   
        self.P_max = 50e-3  
        self.G_WDM = self.P_max / self.B_WDM 
        self.h = 6.626e-34 
        self.nu = 193.1e12 
        self.G_node_dB = 20 
        self.G_node = 10 ** (self.G_node_dB / 10)
        self.span_length = 80.0 
        self.P_ASE_node = 2 * self.h * self.nu * self.n_sp * (self.G_node - 1) * self.B_N

    def compute_arc_noise_C_N(self, length_km):
        num_spans = max(1, int(np.ceil(length_km / self.span_length)))
        actual_span_len = length_km / num_spans
        G_lin_span = 10 ** ((self.alpha_dB * actual_span_len) / 10)
        P_ASE_span = 2 * self.h * self.nu * self.n_sp * (G_lin_span - 1) * self.B_N
        L_eff = (1 - np.exp(-2 * self.alpha * actual_span_len)) / (2 * self.alpha)
        L_eff_a = 1 / (2 * self.alpha)
        asinh_term = np.arcsinh((np.pi**2 / 2) * abs(self.beta2) * L_eff_a * self.B_WDM**2)
        P_NLI_span = (8/27) * (self.gamma**2) * (self.G_WDM**3) * (L_eff**2) * (asinh_term / (np.pi * abs(self.beta2) * L_eff_a)) * self.B_N
        return self.P_ASE_node + num_spans * (P_ASE_span + P_NLI_span)

    def compute_signal_constant_C(self):
        return self.G_WDM * self.B_N

class QoTPreprocessor:
    def __init__(self, G, demands):
        self.G = G
        self.demands = demands
        self.physics = OpticalPhysicsCalculator()

    def combined_preprocessing(self):
        forbidden_arcs = {}
        for u, v, data in self.G.edges(data=True):
            length = data.get('length', 100.0)
            data['cd_weight'] = length * self.physics.d_coeff
            data['osnr_weight'] = self.physics.compute_arc_noise_C_N(length)

        sp_cd = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='cd_weight'))
        sp_osnr = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='osnr_weight'))
        C = self.physics.compute_signal_constant_C()

        for demand in self.demands:
            k = demand['id']
            src, dst = demand['src'], demand['dst']
            max_osnr_noise = (1 / (10 ** (demand['min_osnr'] / 10))) * demand['slots'] * C
            forbidden = set()
            
            for u, v in self.G.edges():
                for a, b in [(u, v), (v, u)]:
                    if sp_cd[src][a] + self.G[a][b]['cd_weight'] + sp_cd[b][dst] > demand['max_cd'] or \
                       sp_osnr[src][a] + self.G[a][b]['osnr_weight'] + sp_osnr[b][dst] > max_osnr_noise:
                        forbidden.add((a, b))
            forbidden_arcs[k] = forbidden
        return forbidden_arcs

def generate_topology(name="NSFNET"):
    G = nx.Graph()
    if name == "SPAIN":
        edges = [(0,1,200), (0,2,350), (1,2,150), (1,3,400), (2,3,250), (2,4,300), (3,4,100)]
    elif name == "COST239":
        edges = [(0,1,300), (0,2,300), (0,3,300), (0,4,300), (0,5,300), (1,2,300), (1,6,300), 
                 (2,3,300), (2,7,300), (3,4,300), (3,8,300), (4,5,300), (4,9,300), (5,6,300), 
                 (5,10,300), (6,7,300), (6,10,300), (7,8,300), (8,9,300), (9,10,300)]
    elif name == "NSFNET":
        edges = [(0,1,1050), (0,2,1500), (0,3,2400), (1,2,600), (1,7,2400), (2,5,1800), 
                 (3,4,600), (3,10,3300), (4,5,1200), (4,6,600), (5,9,1500), (5,12,3000), 
                 (6,7,750), (7,8,750), (8,9,750), (8,11,300), (9,10,1200), (10,12,600), 
                 (10,13,600), (11,12,300), (12,13,300)]
    elif name == "GERMAN":
        edges = [(0,1,200), (0,5,400), (1,2,250), (2,3,320), (3,4,170), (4,5,350), (5,6,270), 
                 (6,7,320), (7,8,180), (8,9,280), (9,10,490), (10,11,390), (11,12,420), 
                 (12,13,500), (13,14,360), (14,15,450), (15,16,210), (16,0,570), (2,7,640), 
                 (4,9,600), (8,13,790), (10,16,1080), (1,11,1200)]
    for u, v, length in edges:
        G.add_edge(u, v, length=length)
    return G

def generate_dynamic_demands(G, n_requests=5):
    nodes = list(G.nodes())
    demands, physics = [], OpticalPhysicsCalculator()
    C = physics.compute_signal_constant_C()
    for u, v, data in G.edges(data=True):
        length = data.get('length', 100.0)
        data['cd_weight'] = length * physics.d_coeff
        data['osnr_weight'] = physics.compute_arc_noise_C_N(length)
        
    profiles = [{'bitrate': 100, 'slots': 3, 'min_osnr': 15, 'max_cd': 40000},
                {'bitrate': 200, 'slots': 4, 'min_osnr': 16, 'max_cd': 20000},
                {'bitrate': 400, 'slots': 6, 'min_osnr': 24, 'max_cd': 20000}]
    sp_cd = dict(nx.all_pairs_dijkstra_path_length(G, weight='cd_weight'))
    sp_osnr = dict(nx.all_pairs_dijkstra_path_length(G, weight='osnr_weight'))
    
    for i in range(n_requests):
        feasible = False
        while not feasible:
            src, dst = random.sample(nodes, 2)
            valid_profiles = [p for p in profiles if sp_cd[src][dst] <= p['max_cd'] and sp_osnr[src][dst] <= (1 / (10 ** (p['min_osnr'] / 10))) * p['slots'] * C]
            if valid_profiles:
                demands.append({'id': i, 'src': src, 'dst': dst, **random.choice(valid_profiles)})
                feasible = True
    return demands

def build_qot_rsa_ilp(G_undirected, demands, total_slots, forbidden_arcs):
    model = Model("QoT-RSA")
    physics = OpticalPhysicsCalculator()
    C = physics.compute_signal_constant_C()
    
    A, arc_lengths = [], {}
    for u, v, data in G_undirected.edges(data=True):
        length = data.get('length', 100.0)
        A.extend([(u, v), (v, u)])
        arc_lengths[(u, v)] = length
        arc_lengths[(v, u)] = length
        
    f = {}
    for demand in demands:
        k, w_k = demand['id'], demand['slots']
        for a in A:
            if a in forbidden_arcs.get(k, set()): continue
            for s in range(w_k - 1, total_slots):
                f[k, a, s] = model.addVar(vtype="B", name=f"f_{k}_{a[0]}_{a[1]}_{s}")
                
    p = model.addVar(vtype="I", name="p", lb=0, ub=total_slots-1)
    model.setObjective(p, "minimize")

    for demand in demands:
        k, o_k, d_k, w_k = demand['id'], demand['src'], demand['dst'], demand['slots']
        src_out = [f[k, (o_k, v), s] for v in G_undirected.neighbors(o_k) for s in range(w_k - 1, total_slots) if (k, (o_k, v), s) in f]
        dst_in = [f[k, (u, d_k), s] for u in G_undirected.neighbors(d_k) for s in range(w_k - 1, total_slots) if (k, (u, d_k), s) in f]
        if src_out: model.addCons(quicksum(src_out) == 1, f"SrcOut_{k}")
        if dst_in: model.addCons(quicksum(dst_in) == 1, f"DstIn_{k}")

        for v in G_undirected.nodes():
            if v != o_k and v != d_k:
                for s in range(w_k - 1, total_slots):
                    in_vars = [f[k, (u, v), s] for u in G_undirected.neighbors(v) if (k, (u, v), s) in f]
                    out_vars = [f[k, (v, x), s] for x in G_undirected.neighbors(v) if (k, (v, x), s) in f]
                    if in_vars or out_vars: model.addCons(quicksum(in_vars) - quicksum(out_vars) == 0)

    for a in A:
        for s_check in range(total_slots):
            overlap = [f[demand['id'], a, s_last] for demand in demands for s_last in range(s_check, min(s_check + demand['slots'], total_slots)) if (demand['id'], a, s_last) in f]
            if overlap: model.addCons(quicksum(overlap) <= 1)
            
    for (k, a, s), var in f.items(): model.addCons(p >= s * var)
    return model, f