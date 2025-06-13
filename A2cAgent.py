
class A2CAgent:
    """
    Advantage Actor-Critic Agent
    """
    def __init__(self, state_size, action_size, actor_lr=0.0002, critic_lr=0.0004, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gradient_accumulation_steps = 8
        self.accumulated_grads = None
        self.accumulation_counter = 0

        # Build actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)


    def _build_actor(self):
        """
        Builds the Actor network (the policy) using a CNN-LSTM architecture.
        This network decides which action to take.
        """
        state_input = Input(shape=self.state_size)

        # 1. Convolutional Layer: Extracts local features from each time step.
        conv_layer = Conv1D(filters=64, kernel_size=3, padding='same')(state_input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)

        # 2. LSTM Layer: Learns long-term temporal dependencies from the sequence of features.
        lstm_layer = LSTM(units=64)(conv_layer)

        # 3. Dense Layer for final processing.
        dense_layer = Dense(64, activation='relu')(lstm_layer)

        # 4. Output Layer: Softmax for action probabilities.
        action_probabilities = Dense(self.action_size, activation='softmax')(dense_layer)

        model = Model(inputs=state_input, outputs=action_probabilities)
        print("--- Actor (Policy) Network with CNN-LSTM Built ---")
        model.summary()
        return model

    def _build_critic(self):
        """
        Builds the Critic network (the value function) using a CNN-LSTM architecture.
        This network estimates how good the current state is.
        """
        state_input = Input(shape=self.state_size)

        # 1. Convolutional Layer (similar to actor for consistent feature extraction).
        conv_layer = Conv1D(filters=64, kernel_size=3, padding='same')(state_input)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('relu')(conv_layer)

        # 2. LSTM Layer to learn temporal dependencies.
        lstm_layer = LSTM(units=64)(conv_layer)

        # 3. Dense Layer for final processing.
        dense_layer = Dense(64, activation='relu')(lstm_layer)

        # 4. Output Layer: A single linear unit for the state value.
        state_value = Dense(1, activation='linear')(dense_layer)

        model = Model(inputs=state_input, outputs=state_value)
        # The critic is compiled with MSE, but training is manual in the 'train' method.
        model.compile(optimizer=Adam(learning_rate=self.critic_lr), loss='mse')
        print("\n--- Critic (Value) Network with CNN-LSTM Built ---")
        model.summary()
        return model

    def act(self, state):
        state_reshaped = np.reshape(state, (1,) + self.state_size)
        action_probs_raw = self.actor.predict(state_reshaped, verbose=0)
        action_probs = np.squeeze(action_probs_raw[0])
        action_probs = np.clip(action_probs, 1e-8, 1.0)
        action_probs /= np.sum(action_probs)
        try:
            action = np.random.choice(self.action_size, p=action_probs)
        except ValueError as e:
            print(f"Error in np.random.choice: {e}")
            print(f"Action Probs Shape: {action_probs.shape}")
            print(f"Action Probs Sum: {np.sum(action_probs)}")
            print(f"Action Probs Values: {action_probs}")
            action = random.randrange(self.action_size)

        return action, action_probs # Return the cleaned probabilities

    # ===== Replace the entire A2CAgent.train method (Cell 4) =====

    def train(self, state, action, reward, next_state, done):
        """Train both actor and critic networks ensuring float32 consistency"""
        # --- Ensure Inputs are Correct Type (Use numpy float32 initially) ---
        reward_np = np.array(reward, dtype=np.float32)
        gamma_np = np.array(self.gamma, dtype=np.float32)

        try:
            state_np = np.reshape(state, (1,) + self.state_size).astype(np.float32)
            next_state_np = np.reshape(next_state, (1,) + self.state_size).astype(np.float32)
        except ValueError as e:
            print(f"Reshape Error: state.shape={state.shape}, self.state_size={self.state_size}, error={e}")
            return 0, 0

        # --- Train Critic ---
        target_value_np = np.array([[reward_np]], dtype=np.float32)
        if not done:
            try:
                next_value_np = self.critic.predict(next_state_np, verbose=0).astype(np.float32)
                if next_value_np.shape != (1, 1): next_value_np = np.reshape(next_value_np, (1, 1)).astype(np.float32)
                target_value_np = np.array([[reward_np]], dtype=np.float32) + gamma_np * next_value_np
            except Exception as e: print(f"Error during critic prediction: {e}")

        target_value_np = np.clip(target_value_np, -100.0, 100.0).astype(np.float32)

        try:
            critic_loss = self.critic.train_on_batch(state_np, target_value_np)
            if np.random.rand() < 0.1: print(f" Critic Update: Loss={critic_loss:.4f}, TargetV={target_value_np[0][0]:.4f}")
        except Exception as e:
            print(f"--- Error during critic.train_on_batch: State shape: {state_np.shape}, Target shape: {target_value_np.shape}, Error: {e}")
            critic_loss = 0

        # --- Train Actor ---
        # Define constants outside the tape
        entropy_coefficient = tf.constant(1.0, dtype=tf.float32) # Keep high entropy, ensure float32
        action_tf = tf.constant(action, dtype=tf.int32)

        # Pre-calculate advantage as numpy float32
        try:
            current_value_np = self.critic.predict(state_np, verbose=0).astype(np.float32)
            if current_value_np.shape != (1, 1): current_value_np = np.reshape(current_value_np, (1, 1)).astype(np.float32)
            advantage_raw_np = target_value_np - current_value_np
            advantage_np = np.clip(advantage_raw_np, -10.0, 10.0).astype(np.float32)
            # Convert to TF Constant here
            advantage_tf_scalar = tf.constant(advantage_np[0, 0], dtype=tf.float32)
            # Debug print
            if np.random.rand() < 0.1: print(f" Advantage Calc: TargetV={target_value_np[0][0]:.4f}, CurrentV={current_value_np[0][0]:.4f}, ClippedAdv={advantage_np[0][0]:.4f}")
        except Exception as e:
             print(f"Error predicting current value / calculating advantage: {e}")
             advantage_tf_scalar = tf.constant(0.0, dtype=tf.float32) # Default scalar float32

        # Convert state to tensor
        state_tf = tf.constant(state_np, dtype=tf.float32)

        with tf.GradientTape() as tape:
            try:
                action_probs = self.actor(state_tf, training=True) # float32

                prob_of_action_taken = tf.gather_nd(action_probs, indices=[[0, action_tf]])
                prob_of_action_taken = tf.clip_by_value(prob_of_action_taken, 1e-10, 1.0)
                log_prob = tf.math.log(prob_of_action_taken) # Shape (1,) float32

                clipped_probs = tf.clip_by_value(action_probs, 1e-10, 1.0) # float32
                entropy = -tf.reduce_sum(clipped_probs * tf.math.log(clipped_probs), axis=1) # Shape (1,) float32

                # <<< --- MOST EXPLICIT CASTING in LOSS CALCULATION --- >>>
                # Cast every term involved right before multiplication/addition
                actor_loss = -tf.cast(log_prob[0], tf.float32) * tf.cast(advantage_tf_scalar, tf.float32) \
                             - tf.cast(entropy_coefficient, tf.float32) * tf.cast(entropy[0], tf.float32)
                # <<< --- END EXPLICIT CASTING --- >>>

            except Exception as actor_e:
                 print(f"Error during actor forward pass or loss calculation: {actor_e}")
                 import traceback
                 traceback.print_exc()
                 actor_loss = tf.constant(0.0, dtype=tf.float32)

        # --- Gradient Calculation and Application ---
        try:
            if tf.is_tensor(actor_loss) and actor_loss.shape == ():
                 actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

                 if actor_grads is None or any(g is None for g in actor_grads):
                    global_norm = tf.linalg.global_norm(actor_grads)
                    # Print for debugging
                    if np.random.rand() < 0.1:
                        print(f"  Global gradient norm: {global_norm.numpy():.6f}")
                    if global_norm < 1e-5:
                        scale_factor = 1e-3 / (global_norm + 1e-10)
                        actor_grads = [g * scale_factor if g is not None else None for g in actor_grads]
                        if np.random.rand() < 0.1:
                            print(f"  Scaling up small gradients by factor: {scale_factor:.6f}")
                    actor_grads = [(tf.clip_by_norm(g, 1.0) if g is not None else None) for g in actor_grads]
                 else:
                     actor_grads = [(tf.clip_by_norm(g, 1.0) if g is not None else None) for g in actor_grads]
                     valid_grads_and_vars = [(g, v) for g, v in zip(actor_grads, self.actor.trainable_variables) if g is not None]
                     if valid_grads_and_vars:
                         self.actor_optimizer.apply_gradients(valid_grads_and_vars)
                         if np.random.rand() < 0.1:
                              grad_means = [tf.reduce_mean(g).numpy() if g is not None else 'None' for g in actor_grads]
                              non_zero_grads = any(m != 0.0 and m != 'None' for m in grad_means)
                              print(f"  Actor Grads Applied (Means): {grad_means} | NonZero: {non_zero_grads}") # Check if non-zero
                     else:
                          if np.random.rand() < 0.1: print("  Actor Grads: None after filtering")
                     actor_loss_np = actor_loss.numpy()
            else:
                 print(f"Warning: Actor loss is not valid: {actor_loss}")
                 actor_loss_np = 0
        except Exception as grad_e:
             print(f"Error during gradient calculation/application: {grad_e}")
             import traceback
             traceback.print_exc()
             actor_loss_np = 0

        return critic_loss, actor_loss_np # Return scalar losses

# ===== END of Replacement =====

def train_a2c_agent(env, agent, episodes=45):
    """Train an A2C agent on the environment"""
    rewards = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, action_probs = agent.act(state)

            step_num_in_episode = env.current_date_idx - env.window_size # Estimate step number
            if step_num_in_episode % 50 == 0: # Print every 50 steps within episode
                print(f"Episode {e+1}, Step {step_num_in_episode}: Action Chosen: {action}, Prob[Action]: {action_probs[action]:.4f}")

            next_state, reward, done, info = env.step(action)

            # Train on this experience
            agent.train(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode: {e+1}/{episodes}, Reward: {total_reward}")
                rewards.append(total_reward)

    return rewards