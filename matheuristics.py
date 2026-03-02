import numpy as np
import networkx as nx
from pyscipopt import Model, quicksum

# ==========================================
# 2. SOTA MATHEURISTICS (NO ABSM POISON)
# ==========================================
class RSAMatheuristics:
    def __init__(self, demands, network, total_slots, forbidden_arcs):
        self.demands = {d['id']: d for d in demands}
        self.network = network
        self.total_slots = total_slots
        self.forbidden_arcs = forbidden_arcs
        self.routes_dict = self._precompute_routes(k_shortest=10)

    def _precompute_routes(self, k_shortest):
        routes = {}
        for d_id, demand in self.demands.items():
            G_valid = nx.DiGraph(self.network)
            for u, v in self.forbidden_arcs.get(d_id, set()):
                if G_valid.has_edge(u, v): G_valid.remove_edge(u, v)
            try:
                paths = list(nx.shortest_simple_paths(G_valid, demand['src'], demand['dst'], weight='length'))[:k_shortest]
                routes[d_id] = [[(p[i], p[i+1]) for i in range(len(p)-1)] for p in paths]
            except nx.NetworkXNoPath:
                routes[d_id] = []
        return routes

    def _is_valid_absm_slot(self, s, req, usage_matrix, route, arc_to_idx):
        if s == 0: return True
        for u, v in route:
            arc_idx = arc_to_idx[(u, v)]
            if s > 0 and usage_matrix[arc_idx, s - 1]:
                return True
        return False

    def _fast_graph_packer(self, ordering, strategy='R-SA', use_absm=False):
        usage = np.zeros((len(self.network.edges()) * 2, self.total_slots), dtype=bool)
        arcs = list(self.network.edges())
        arc_to_idx = {arc: i for i, arc in enumerate(arcs + [(v, u) for u, v in arcs])}
        sol_dict = {}
        
        for d_id in ordering:
            req = self.demands[d_id]['slots']
            routed = False
            for route in self.routes_dict[d_id]:
                for s in range(self.total_slots - req + 1):
                    if use_absm and not self._is_valid_absm_slot(s, req, usage, route, arc_to_idx):
                        continue
                        
                    if not any(np.any(usage[arc_to_idx[arc], s:s+req]) for arc in route):
                        for u, v in route:
                            usage[arc_to_idx[(u, v)], s:s+req] = True
                            sol_dict[f"f_{d_id}_{u}_{v}_{s}"] = 1.0 
                        routed = True
                        break
                if routed: break
                
            if not routed: 
                if use_absm:
                    return self._fast_graph_packer(ordering, strategy, use_absm=False)
                else:
                    return None 
        return sol_dict

    def _extract_sub_mip_dict(self, sub_model):
        if sub_model.getNSols() > 0:
            sol_dict = {}
            best_sub_sol = sub_model.getBestSol()
            for sub_var in sub_model.getVars():
                val = sub_model.getSolVal(best_sub_sol, sub_var)
                if val > 0.5: sol_dict[sub_var.name] = val
            return sol_dict
        return None

    def run_neural_diving(self, main_model, var_assignments, top_k_percent=0.05):
        sub_model = Model(sourceModel=main_model)
        # Academic SOTA: Allow MILP heuristics 15s to discover neighborhood bounds
        sub_model.setRealParam("limits/time", 15.0) 
        sub_model.hideOutput()
        vars_dict = {v.name: v for v in sub_model.getVars()}
        
        valid_vars = {name: prob for name, prob in var_assignments.items() if name in vars_dict and name.startswith('f_')}
        if not valid_vars: return None
        
        sorted_vars = sorted(valid_vars.items(), key=lambda item: abs(item[1] - 0.5), reverse=True)
        num_to_fix = max(1, int(len(sorted_vars) * top_k_percent))
        
        for var_name, prob in sorted_vars[:num_to_fix]:
            parts = var_name.split('_')
            if len(parts) == 5:
                s_idx = int(parts[4])
                if s_idx % 2 != 0: prob = prob * 0.5 
            
            if prob > 0.5: sub_model.fixVar(vars_dict[var_name], 1.0)
            else: sub_model.fixVar(vars_dict[var_name], 0.0)
                
        sub_model.optimize()
        return self._extract_sub_mip_dict(sub_model)

    def run_local_branching(self, main_model, k_size=5):
        if main_model.getNSols() == 0: return None 
        sub_model = Model(sourceModel=main_model)
        sub_model.setRealParam("limits/time", 15.0) 
        sub_model.hideOutput()
        
        best_sol = main_model.getBestSol()
        sub_vars = sub_model.getVars()
        main_vars = {v.name: v for v in main_model.getVars(transformed=True)}
        
        hamming_expr = [1 - s_var if main_model.getSolVal(best_sol, main_vars[s_var.name]) > 0.5 else s_var for s_var in sub_vars if s_var.name in main_vars]
        if hamming_expr:
            sub_model.addCons(quicksum(hamming_expr) <= k_size)
            
        sub_model.optimize()
        return self._extract_sub_mip_dict(sub_model)