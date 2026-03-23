import os
import pandas as pd

from hardware_setup import MACHINE_SPEED_FACTOR
from rl_components import train_success_oracle, run_exhaustive_benchmarks
from plotting_utils import generate_exhaustive_report

# ==========================================
# 7. EXECUTE EVERYTHING
# ==========================================
if __name__ == "__main__":
    TRAIN_TOPOLOGIES = ["SPAIN", "NSFNET", "COST239"]
    TEST_TOPOLOGIES = ["GERMAN"]
    NUM_SLOTS = 60 # Safe memory threshold for 180s timeout 
    
    EPISODES = 500 
    ARCHITECTURES = ["tripartite", "bipartite"]
    
    SAVE_DIRECTORY = "latex_assets"
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)
    
    for arch in ARCHITECTURES:
        print(f"\n{'='*60}")
        print(f"=== STARTING FULL SCALED PIPELINE: {arch.upper()} + SOTA ===")
        print(f"{'='*60}\n")
        
        print(f"1. Training {arch.capitalize()} Agent for {EPISODES} episodes...")
        history_df, trained_driver = train_success_oracle(
            architecture=arch, topologies=TRAIN_TOPOLOGIES, episodes=EPISODES, save_dir=SAVE_DIRECTORY 
        )
        
        print(f"\n2. Running Benchmarks on Unseen Test Set {TEST_TOPOLOGIES}...")
        df_res, conv_data, actions = run_exhaustive_benchmarks(
            trained_driver=trained_driver, num_slots=NUM_SLOTS, 
            topologies=TRAIN_TOPOLOGIES+TEST_TOPOLOGIES, architecture=arch
        )
        
        print(f"\n3. Generating LaTeX Report and PNG Figures...")
        generate_exhaustive_report(
            history_df=history_df, df_results=df_res, convergence_data=conv_data, 
            action_counts=actions, architecture=arch, save_dir=SAVE_DIRECTORY
        )
        
        print(f"\nFinished scaled pipeline. Assets saved to ./{SAVE_DIRECTORY}/")