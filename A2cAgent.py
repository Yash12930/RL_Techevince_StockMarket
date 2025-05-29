import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent for portfolio management.
    """
    def __init__(self, state_size, action_size, hidden_units=128, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.gamma = 0.95

        self.actor, self.critic = self._build_model()

        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []

    def _build_model(self):
        """
        Builds actor and critic networks with shared CNN + LSTM feature extraction.
        """
        state_input = Input(shape=self.state_size)

        x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(state_input)
        x = Conv1D(128, kernel_size=5, padding='same', activation='relu')(x)
        x = LSTM(self.hidden_units, return_sequences=True)(x)
        x = LSTM(self.hidden_units)(x)

        actor_x = Dropout(0.3)(Dense(self.hidden_units, activation='relu')(x))
        actor_output = Dense(self.action_size, activation='softmax')(actor_x)

        critic_x = Dropout(0.3)(Dense(self.hidden_units, activation='relu')(x))
        critic_output = Dense(1, activation='linear')(critic_x)

        actor = Model(inputs=state_input, outputs=actor_output)
        critic = Model(inputs=state_input, outputs=critic_output)

        actor.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return actor, critic

    def act(self, state):
        """
        Selects an action using the actor network.
        """
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor.predict(state, verbose=0)[0]

        if np.random.rand() < 0.8:
            action = np.random.choice(self.action_size, p=action_probs)
        else:
            action = np.argmax(action_probs)

        return action, action_probs

    def train(self, states, actions, rewards, next_states, dones):
        """
        Updates actor and critic networks using A2C.
        """
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        values = self.critic.predict(states, verbose=0).flatten()
        next_values = self.critic.predict(next_states, verbose=0).flatten()

        returns = rewards + self.gamma * next_values * (1 - dones)
        advantages = returns - values

        action_onehot = tf.one_hot(actions, self.action_size)

        critic_loss = self.critic.train_on_batch(states, returns)

        with tf.GradientTape() as tape:
            action_probs = self.actor(states, training=True)
            selected_probs = tf.reduce_sum(action_probs * action_onehot, axis=1)
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
            actor_loss = -tf.reduce_mean(tf.math.log(selected_probs + 1e-10) * advantages + 0.01 * entropy)

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        self.actor_losses.append(float(actor_loss))
        self.critic_losses.append(critic_loss)

        return float(actor_loss), critic_loss

    def save(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
