import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os

FOLDER = './latex_assets/latest/'

def plot_smoothed_training(csv_file, save_name, title_prefix):
    csv_file = FOLDER+csv_file
    save_name = FOLDER+save_name
    
    if not os.path.exists(csv_file):
        print(f"Could not find {csv_file}")
        return
        
    df = pd.read_csv(csv_file)
    window_size = 30 # Matches the 30-scenario cycle
    
    df['agent_smooth'] = df['agent_pio'].rolling(window=window_size, min_periods=1).mean()
    df['default_smooth'] = df['default_pio'].rolling(window=window_size, min_periods=1).mean()

    plt.figure(figsize=(8, 5))
    
    # Faint background noise (raw data)
    plt.plot(df['episode'], df['agent_pio'], color='#3498db', alpha=0.15)
    plt.plot(df['episode'], df['default_pio'], color='#e74c3c', alpha=0.15)
    
    # Thick foreground lines (Moving average)
    plt.plot(df['episode'], df['agent_smooth'], label=f'{title_prefix} Agent (30-ep Avg)', color='#2980b9', linewidth=2.5)
    plt.plot(df['episode'], df['default_smooth'], linestyle='--', color='#c0392b', label='Default SCIP (30-ep Avg)', linewidth=2.5)
    
    plt.title(f"{title_prefix} Training Convergence (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Primal Integral (PIO)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=150)
    print(f"Saved {save_name}.png")
    plt.close()

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

def plot_all_topologies():
    # Ensure the output directory exists for your LaTeX draft
    os.makedirs("latex_assets", exist_ok=True)
    
    topologies = ["SPAIN", "NSFNET", "COST239", "GERMAN"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, topo_name in enumerate(topologies):
        ax = axes[i]
        G = generate_topology(topo_name)
        
        # Use a spring layout or spectral layout for aesthetics
        pos = nx.spring_layout(G, seed=42) 
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='#3498db', 
                               node_size=600, edgecolors='black', linewidths=1.5)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', width=1.5, alpha=0.7)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold', font_color='white')
        
        # Draw edge labels (lengths)
        edge_labels = nx.get_edge_attributes(G, 'length')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8)
        
        # Dynamic Titles
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        title_str = f"{topo_name} Network\n({num_nodes} Nodes, {num_edges} Links)"
        
        # Mark GERMAN distinctly as the unseen testing topology
        if topo_name == "GERMAN":
            title_str += "\n[Unseen Zero-Shot Test Set]"
            
        ax.set_title(title_str, fontsize=14, fontweight="bold", pad=10)
        ax.axis('off') # Hide the bounding box for a cleaner look

    plt.tight_layout(pad=3.0)
    
    # Save as high-resolution PNG
    save_path = os.path.join("latex_assets", "topology_examples.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated high-resolution plot at: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Generate plots for Version 3 Data
    plot_smoothed_training('bipartite_training_history.csv', 'bipartite_smoothed_curve', 'Bipartite')
    plot_smoothed_training('tripartite_training_history.csv', 'tripartite_smoothed_curve', 'Tripartite')
    plot_all_topologies()