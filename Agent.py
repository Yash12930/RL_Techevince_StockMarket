import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam




def train_a2c_agent(env, agent, episodes=100, max_steps=1000):
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0

        states, actions, rewards_, next_states, dones = [], [], [], [], []

        for _ in range(max_steps):
            action, _ = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards_.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            ep_reward += reward

            if done:
                break

        actor_loss, critic_loss = agent.train(states, actions, rewards_, next_states, dones)
        rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            print(f"[A2C] Episode {ep+1}, Reward: {ep_reward:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

        if (ep + 1) % 50 == 0:
            agent.save(f"a2c_actor_ep{ep+1}.h5", f"a2c_critic_ep{ep+1}.h5")

    return rewards
