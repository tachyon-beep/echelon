
import numpy as np
from echelon import EchelonEnv, EnvConfig

def test_obs_hull_onehot():
    cfg = EnvConfig(num_packs=1, seed=0)
    env = EchelonEnv(cfg)
    obs, _ = env.reset(seed=0)
    
    # self_dim = 40
    # acoustic(4) + hull(4) + status(32)
    # The acoustic is at the end of the ego parts, but let's check the _obs_dim
    
    total_dim = env._obs_dim()
    self_dim = 40
    self_start = total_dim - self_dim
    
    for aid, o in obs.items():
        viewer = env.sim.mechs[aid]
        hull_name = viewer.spec.name
        
        # Hull one-hot is after 4 acoustic quadrants
        hull_onehot = o[self_start + 4 : self_start + 8]
        
        hull_map = {"scout": 0, "light": 1, "medium": 2, "heavy": 3}
        expected_idx = hull_map[hull_name]
        
        expected = np.zeros(4)
        expected[expected_idx] = 1.0
        
        print(f"Agent {aid} ({hull_name}): obs={hull_onehot}, expected={expected}")
        assert np.allclose(hull_onehot, expected), f"Hull one-hot mismatch for {aid}"
        
    print("One-hot hull observation test passed!")

if __name__ == "__main__":
    test_obs_hull_onehot()
