'''
this is where the agent sits
'''
import numpy as np
from engine import Tensor
from nlp import get_sentiment_vector
from env import TradingEnvironment
from engine import Tensor
from Neural_Nets import LayerNorm, Dropout, Linear, Adam_Optimiser

class PPOAgent:
    def __init__(self, state_size = 67, action_size = 2):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.gamma = 0.99
        self.epsilon = 0.2
        self.epochs = 5
        self.std = 0.3
        self.std_min = 0.05
        self.std_decay = 0.995

        self.actor_l1 = Linear(state_size, 64)
        self.actor_norm1 = LayerNorm(64)
        self.actor_drop1 = Dropout(0.1)
        
        self.actor_l2 = Linear(64 , 32)
        self.actor_norm2 = LayerNorm(32)
        self.actor_drop2 = Dropout(0.1)
        
        self.actor_out = Linear(32, action_size)

        self.critic_l1 = Linear(state_size, 64)
        self.critic_norm1 = LayerNorm(64)
        self.critic_drop1 = Dropout(0.1)
        
        self.critic_l2 = Linear(64, 32)
        self.critic_norm2 = LayerNorm(32)
        self.critic_drop2 = Dropout(0.1)
        
        self.critic_out = Linear(32, 1)

        all_params = []
        all_params.extend(self.actor_l1.parameters())
        all_params.extend(self.actor_norm1.parameters())
        all_params.extend(self.actor_l2.parameters())
        all_params.extend(self.actor_norm2.parameters())
        all_params.extend(self.actor_out.parameters())
        
        all_params.extend(self.critic_l1.parameters())
        all_params.extend(self.critic_norm1.parameters())
        all_params.extend(self.critic_l2.parameters())
        all_params.extend(self.critic_norm2.parameters())
        all_params.extend(self.critic_out.parameters())
        
        self.optimizer = Adam_Optimiser(all_params, lr=0.0003)
        
    def select_action(self , state):
        self.actor_drop1.training = False
        self.actor_drop2.training = False

        x = self.actor_drop1(self.actor_norm1(self.actor_l1(state).relu()))
        x = self.actor_drop2(self.actor_norm2(self.actor_l2(x).relu()))
        out = self.actor_out(x)

        direction_mean = out[0].tanh()
        size_mean = out[1].sigmoid()

        direction = direction_mean.data + np.random.normal(0, self.std)
        size = size_mean.data + np.random.normal(0, self.std)
        direction = np.clip(direction, -1, 1)
        size = np.clip(size, 0, 1)    

        log_prob_d = -0.5 * ((direction - direction_mean.data) / self.std) ** 2
        log_prob_s = -0.5 * ((size - size_mean.data) / self.std) ** 2
        log_prob = log_prob_d + log_prob_s

        self.critic_drop1.training = False
        self.critic_drop2.training = False
        v = self.critic_drop1(self.critic_norm1(self.critic_l1(state).relu()))
        v = self.critic_drop2(self.critic_norm2(self.critic_l2(v).relu()))
        value = self.critic_out(v)

        self.states.append(state)
        self.actions.append(np.array([direction, size]))
        self.log_probs.append(log_prob)
        self.values.append(value.data[0])

        self.actor_drop1.training = True
        self.actor_drop2.training = True
        self.critic_drop1.training = True
        self.critic_drop2.training = True
        
        return np.array([direction, size])
    
    def compute_rewards(self , next_value):
        rewards = []
        R = next_value
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = np.array(rewards) 
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards
    
    def update(self):
        # Step 1 — get stored data
        states = self.states
        actions = self.actions
        old_log_probs = np.array(self.log_probs)
        
        # Step 2 — compute returns and advantages
        returns = self.compute_rewards(next_value=0)
        advantages = returns - np.array(self.values)

        # Step 3 — PPO update loop
        for _ in range(self.epochs):
            for i in range(len(states)):
                state = states[i]
                action = actions[i]
                old_log_p = old_log_probs[i]
                adv = advantages[i]
                ret = returns[i]

                x = self.actor_drop1(self.actor_norm1(self.actor_l1(state).relu()))
                x = self.actor_drop2(self.actor_norm2(self.actor_l2(x).relu()))
                out = self.actor_out(x)
                
                direction_mean = out[0].tanh()
                size_mean = out[1].sigmoid()
                new_log_prob_d = -0.5 * ((action[0] - direction_mean.data) / self.std) ** 2
                new_log_prob_s = -0.5 * ((action[1] - size_mean.data) / self.std) ** 2
                new_log_prob   = new_log_prob_d + new_log_prob_s  

                ratio_data = np.exp(new_log_prob.data - old_log_p)
                clipped    = np.clip(ratio_data, 1 - self.epsilon, 1 + self.epsilon)

                surr1 = Tensor(ratio_data * adv)
                surr2 = Tensor(clipped * adv)
                actor_loss = Tensor(-np.minimum(surr1.data, surr2.data))

                # Critic forward with Norm and Dropout
                v = self.critic_drop1(self.critic_norm1(self.critic_l1(state).relu()))
                v = self.critic_drop2(self.critic_norm2(self.critic_l2(v).relu()))
                new_value = self.critic_out(v)
                
                critic_loss = (Tensor(np.array([ret])) - new_value) ** 2

                loss = (actor_loss + 0.5 * critic_loss).sum()

                # Step 4 — backward + step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Step 5 — decay std + clear memory
        self.std = max(self.std_min, self.std * self.std_decay)
        self.states, self.actions, self.rewards, self.log_probs, self.values = [], [], [], [], []
