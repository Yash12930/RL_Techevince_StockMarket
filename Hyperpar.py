from sklearn.model_selection import ParameterGrid
import json
from datetime import datetime

class HyperparameterTuner:
    """
    Hyperparameter tuning for reinforcement learning agents
    """
    def __init__(self, env_creator, agent_class, param_grid):
        self.env_creator = env_creator  # Function that creates environment
        self.agent_class = agent_class  # Agent class
        self.param_grid = param_grid    # Dictionary of parameter ranges
        self.results = []

    def run_experiment(self, params, episodes=50):
        """Run a single experiment with given parameters"""
        # Create environment
        env = self.env_creator()

        # Create agent with specific parameters
        agent = self.agent_class(**params)

        # Training loop
        rewards = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                if hasattr(agent, 'act'):
                    if agent_class.__name__ == 'PPOAgent' or agent_class.__name__ == 'A2CAgent':
                        action, _ = agent.act(state)
                    else:
                        action = agent.act(state)
                else:
                    # Fallback for different agent interfaces
                    action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action)

                # Different agent interfaces
                if hasattr(agent, 'remember'):
                    agent.remember(state, action, reward, next_state, done)

                if hasattr(agent, 'replay'):
                    agent.replay()

                state = next_state
                total_reward += reward

            rewards.append(total_reward)

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{episodes}, Reward: {total_reward:.2f}")

        # Calculate performance metrics
        avg_reward = sum(rewards) / len(rewards)
        max_reward = max(rewards)
        last_10_avg = sum(rewards[-10:]) / 10

        return {
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'last_10_avg': last_10_avg,
            'rewards': rewards
        }

    def tune(self, experiments_per_config=3, episodes_per_experiment=50):
        """Run hyperparameter tuning"""
        all_params = list(ParameterGrid(self.param_grid))
        print(f"Testing {len(all_params)} parameter configurations...")

        for i, params in enumerate(all_params):
            print(f"\nConfiguration {i+1}/{len(all_params)}: {params}")

            # Run multiple experiments for statistical significance
            config_results = []
            for j in range(experiments_per_config):
                print(f"  Experiment {j+1}/{experiments_per_config}")
                result = self.run_experiment(params, episodes=episodes_per_experiment)
                config_results.append(result)

            # Aggregate results
            avg_over_experiments = {
                'avg_reward': sum(r['avg_reward'] for r in config_results) / len(config_results),
                'max_reward': sum(r['max_reward'] for r in config_results) / len(config_results),
                'last_10_avg': sum(r['last_10_avg'] for r in config_results) / len(config_results),
            }

            # Store results
            self.results.append({
                'params': params,
                'results': avg_over_experiments,
                'raw_results': config_results
            })

        # Sort by average reward
        self.results.sort(key=lambda x: x['results']['avg_reward'], reverse=True)

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"hyperparameter_tuning_{timestamp}.json", 'w') as f:
            json.dump(self.results, f, indent=4)

        return self.results

    def get_best_params(self):
        """Get the best parameters from tuning results"""
        if not self.results:
            return None
        return self.results[0]['params']

