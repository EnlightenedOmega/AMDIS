import sys
import os

from server.microbiome_environment import MicrobiomeEnvironment
from models import MicrobiomeAction

def test_environment():
    env = MicrobiomeEnvironment()
    obs = env.reset()
    
    print(f"Initial Health: {obs.health_marker:.2f}, Drug: {obs.drug_concentration:.2f}, Metabolite: {obs.metabolite_concentration:.2f}")
    print(f"Initial Abundances: {[round(x, 2) for x in obs.microbiome_abundances]}")
    
    for i in range(50):
        # Administer a 1.5 dose every 10 steps
        dosage = 1.5 if i % 10 == 0 else 0.0
        action = MicrobiomeAction(dosage=dosage)
        obs = env.step(action)
        
        print(f"Step {env.state.step_count:02d} | Dose: {dosage:.1f} | Health: {obs.health_marker:.2f} | Drug: {obs.drug_concentration:.2f} | Met: {obs.metabolite_concentration:.2f} | Reward: {obs.reward:.2f}")
        
        if obs.done:
            print(f"Episode done early at step {env.state.step_count}!")
            break

if __name__ == "__main__":
    test_environment()
