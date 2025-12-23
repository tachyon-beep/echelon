import pytest
import torch

from echelon import EchelonEnv, EnvConfig
from echelon.config import WorldConfig
from echelon.rl.model import ActorCriticLSTM


def test_ppo_training_loop_integration():
    """
    Verify that a PPO training loop can run for a few iterations without
    crashing. This tests the integration of Env, Model, and Optimizer.
    """
    device = "cpu"
    cfg = EnvConfig(
        world=WorldConfig(size_x=20, size_y=20, size_z=10),
        num_packs=1,
        seed=42,
        max_episode_seconds=5.0,
        observation_mode="full",
    )
    env = EchelonEnv(cfg)

    # Initialize model
    dummy_obs, _ = env.reset(seed=42)
    sample_agent = env.agents[0]
    obs_dim = dummy_obs[sample_agent].shape[0]
    action_dim = env.ACTION_DIM

    # Use LSTM model from codebase
    model = ActorCriticLSTM(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run a tiny training loop (collect -> train)
    steps = 20
    batch_size = len(env.agents)

    obs = dummy_obs
    lstm_state = model.initial_state(batch_size, device=torch.device(device))
    done = torch.zeros(batch_size, device=device)  # No resets in this short loop

    # Collection phase
    try:
        log_probs_list = []

        for _ in range(steps):
            # Batch inference for all agents
            agent_ids = env.agents
            obs_tensor = torch.stack([torch.from_numpy(obs[aid]) for aid in agent_ids]).to(device)

            # Forward pass
            action, logprob, _, _value, next_lstm_state = model.get_action_and_value(
                obs_tensor, lstm_state, done
            )

            actions_np = action.detach().cpu().numpy()
            act_dict = {aid: actions_np[i] for i, aid in enumerate(agent_ids)}

            next_obs, _rewards, term, trunc, _ = env.step(act_dict)

            # Store data (simplified for test)
            log_probs_list.append(logprob)

            obs = next_obs
            lstm_state = next_lstm_state  # Carry over state
            # For this test, we ignore done/reset logic for simplicity in state handling
            # because steps=20 is < max_episode_seconds=5.0 (which is 50 steps at dt=0.1)

            if any(term.values()) or any(trunc.values()):
                obs, _ = env.reset()
                lstm_state = model.initial_state(batch_size, device=torch.device(device))

        # Optimization phase (dummy backward pass)
        loss = torch.tensor(0.0, requires_grad=True)
        # Just check gradients can be computed
        for lp in log_probs_list:
            loss = loss + lp.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    except Exception as e:
        pytest.fail(f"Training loop crashed: {e}")

    print("PPO integration test passed.")
