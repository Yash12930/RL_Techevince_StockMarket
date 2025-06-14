from sklearn.model_selection import ParameterGrid
import json
from datetime import datetime

class HyperparameterTuner:
    def __init__(self, env_creator, agent_class, param_grid, static_params=None):
        self.env_creator = env_creator
        self.agent_class = agent_class
        self.param_grid = param_grid
        self.results = []
        # Store static parameters, defaulting to an empty dict if None
        self.static_params = static_params if static_params is not None else {}

    def run_experiment(self, params, episodes=50):
        """Run a single experiment with a given set of hyperparameters."""
        env = self.env_creator()

        # <<<< FIX: Combine static params with the current hyperparameter set >>>>
        full_params = self.static_params.copy()
        full_params.update(params)

        # Create agent with the complete set of parameters
        agent = self.agent_class(**full_params)
        # <<<< END FIX >>>>

        rewards = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                # Handle agents that return action_probs and those that don't
                act_result = agent.act(state)
                action = act_result[0] if isinstance(act_result, tuple) else act_result

                next_state, reward, done, _ = env.step(action)

                # Train the agent
                if hasattr(agent, 'train'):
                    agent.train(state, action, reward, next_state, done)
                elif hasattr(agent, 'remember'):
                    agent.remember(state, action, reward, next_state, done)
                    if hasattr(agent, 'replay'):
                        agent.replay()

                state = next_state
                total_reward += reward

            rewards.append(total_reward)
            if (episode + 1) % 10 == 0:
                print(f"  ... Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        return {'avg_reward': avg_reward, 'rewards': rewards}

    def tune(self, experiments_per_config=1, episodes_per_experiment=50):
        """Run the full hyperparameter tuning process."""
        all_params = list(ParameterGrid(self.param_grid))
        print(f"Testing {len(all_params)} parameter configurations...")

        for i, params in enumerate(all_params):
            print(f"\nConfiguration {i+1}/{len(all_params)}: {params}")
            config_results = []
            for j in range(experiments_per_config):
                print(f"  Experiment {j+1}/{experiments_per_config}:")
                result = self.run_experiment(params, episodes=episodes_per_experiment)
                config_results.append(result)

            avg_reward_for_config = sum(r['avg_reward'] for r in config_results) / len(config_results)
            self.results.append({'params': params, 'avg_reward': avg_reward_for_config})

        self.results.sort(key=lambda x: x['avg_reward'], reverse=True)
        return self.results

    def get_best_params(self):
        """Get the best parameters from the tuning results."""
        if not self.results:
            return None
        return self.results[0]['params']

import json
from sklearn.model_selection import ParameterGrid

# <<<< REPLACE THIS ENTIRE FUNCTION >>>>

def tune_a2c_hyperparameters(env_creator, episodes_per_experiment=10):
    """
    Tune A2C agent hyperparameters. This version correctly passes state_size
    and action_size to the tuner.
    """
    print("\n" + "="*20 + " Starting A2C Hyperparameter Tuning " + "="*20)

    # --- FIX: Get state and action sizes BEFORE creating the tuner ---
    # Create a temporary environment just to get the dimensions
    print("Creating temporary environment to determine state and action sizes...")
    temp_env = env_creator()
    state_size = temp_env.observation_space.shape
    action_size = temp_env.action_space.n
    print(f"Determined state_size: {state_size}, action_size: {action_size}")

    # Prepare the dictionary of static (non-tuned) parameters
    static_params = {
        'state_size': state_size,
        'action_size': action_size
    }
    # --- END FIX ---

    param_grid = {
        'actor_lr': [0.0001, 0.0005],
        'critic_lr': [0.0002, 0.0008],
        'gamma': [0.95, 0.99]
    }

    # --- FIX: Pass the static_params to the tuner ---
    tuner = HyperparameterTuner(
        env_creator,
        A2CAgent,
        param_grid,
        static_params=static_params
    )
    # --- END FIX ---

    print(f"Tuning with {len(list(ParameterGrid(param_grid)))} parameter combinations...")
    best_results = tuner.tune(experiments_per_config=1, episodes_per_experiment=episodes_per_experiment)

    best_params = tuner.get_best_params()
    print("\n" + "="*20 + " Hyperparameter Tuning Complete " + "="*20)

    if best_params:
        print("\nBest Parameters Found:")
        print(json.dumps(best_params, indent=4))
    else:
        print("\nTuning did not yield conclusive results. Using default parameters.")
        best_params = {'actor_lr': 0.0002, 'critic_lr': 0.0004, 'gamma': 0.99}

    return best_params
