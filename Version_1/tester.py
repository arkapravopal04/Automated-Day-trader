from agent import PPOAgent
from engine import Tensor
import numpy as np

agent = PPOAgent(state_size=67, action_size=2)
fake_state = Tensor(np.random.randn(67))
action = agent.select_action(fake_state)
print(f"Action: {action}")
print(f"Direction: {action[0]:.3f}  Size: {action[1]:.3f}")
agent.rewards.append(0.05)
print("Agent working ✓")