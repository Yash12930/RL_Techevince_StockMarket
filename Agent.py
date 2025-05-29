import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class PPOAgent:

    def __init__(self, state_size, action_size, hidden_units=128, learning_rate=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_ratio = 0.2
        self.epochs = 10

        self.model = self._build_model()
        self.losses = []

    def _build_model(self):
        state_input = Input(shape=self.state_size)

        x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(state_input)
        x = Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)

        actor_x = Flatten()(Conv1D(64, 3, activation='relu')(x))
        actor_output = Dense(self.action_size, activation='softmax', name='actor_output')(Dropout(0.3)(Dense(self.hidden_units, activation='relu')(actor_x)))

        critic_x = Flatten()(Conv1D(64, 3, activation='relu')(x))
        critic_output = Dense(1, activation='linear', name='critic_output')(Dropout(0.3)(Dense(self.hidden_units, activation='relu')(critic_x)))

        model = Model(inputs=state_input, outputs=[actor_output, critic_output])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={'actor_output': self._ppo_loss, 'critic_output': 'mse'},
            loss_weights={'actor_output': 1.0, 'critic_output': 0.5}
        )

        return model

    def _ppo_loss(self, y_true, y_pred):
        advantages = y_true[:, :1]
        old_probs = y_true[:, 1:]
        new_probs = y_pred

        ratio = new_probs / (old_probs + 1e-10)
        p1 = ratio * advantages
        p2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        entropy = -tf.reduce_mean(new_probs * tf.math.log(new_probs + 1e-10))
        return -tf.reduce_mean(tf.minimum(p1, p2)) - 0.01 * entropy

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs, _ = self.model.predict(state, verbose=0)
        action = np.random.choice(self.action_size, p=action_probs[0])
        return action, action_probs[0]

    def train(self, states, actions, rewards, next_states, dones, old_probs):
        states, next_states = np.array(states), np.array(next_states)
        actions, rewards, dones = np.array(actions), np.array(rewards), np.array(dones)
        old_probs = np.array(old_probs)

        action_probs, values = self.model.predict(states, verbose=0)
        _, next_values = self.model.predict(next_states, verbose=0)

        old_action_probs = np.zeros_like(action_probs)
        for i, (a, p) in enumerate(zip(actions, old_probs)):
            old_action_probs[i, a] = p

        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            next_val = 0 if dones[t] else (next_values[t][0] if t == len(rewards) - 1 else values[t+1][0])
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t][0]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t][0]

        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        actor_target = np.concatenate([advantages[:, None], old_action_probs], axis=-1)

        history = self.model.fit(
            states,
            {'actor_output': actor_target, 'critic_output': returns},
            epochs=self.epochs,
            batch_size=64,
            verbose=0
        )

        self.losses.append(history.history['loss'][-1])
        return history.history['loss'][-1]

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)


def train_ppo_agent(env, agent, episodes=100, max_steps=1000):
    rewards = []
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0

        states, actions, rewards_, next_states, dones, old_probs = [], [], [], [], [], []

        for _ in range(max_steps):
            action, probs = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards_.append(reward)
            next_states.append(next_state)
            dones.append(done)
            old_probs.append(probs[action])

            state = next_state
            ep_reward += reward

            if done:
                break

        loss = agent.train(states, actions, rewards_, next_states, dones, old_probs)
        rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            print(f"[PPO] Episode {ep+1}, Reward: {ep_reward:.2f}, Loss: {loss:.4f}")

        if (ep + 1) % 50 == 0:
            agent.save(f"ppo_agent_ep{ep+1}.h5")

    return rewards


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
