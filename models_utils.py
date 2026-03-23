'''
Handles saving and loading model weights
'''

import pickle
import os
import numpy as np

def save_model(agent, filepath):
    weights = {
        # actor
        'actor_l1_W':    agent.actor_l1.W.data,
        'actor_l1_b':    agent.actor_l1.b.data,
        'actor_norm1_g': agent.actor_norm1.gamma.data,
        'actor_norm1_b': agent.actor_norm1.beta.data,
        'actor_l2_W':    agent.actor_l2.W.data,
        'actor_l2_b':    agent.actor_l2.b.data,
        'actor_norm2_g': agent.actor_norm2.gamma.data,
        'actor_norm2_b': agent.actor_norm2.beta.data,
        'actor_out_W':   agent.actor_out.W.data,
        'actor_out_b':   agent.actor_out.b.data,
        # critic
        'critic_l1_W':    agent.critic_l1.W.data,
        'critic_l1_b':    agent.critic_l1.b.data,
        'critic_norm1_g': agent.critic_norm1.gamma.data,
        'critic_norm1_b': agent.critic_norm1.beta.data,
        'critic_l2_W':    agent.critic_l2.W.data,
        'critic_l2_b':    agent.critic_l2.b.data,
        'critic_norm2_g': agent.critic_norm2.gamma.data,
        'critic_norm2_b': agent.critic_norm2.beta.data,
        'critic_out_W':   agent.critic_out.W.data,
        'critic_out_b':   agent.critic_out.b.data,
        # training state
        'std':            agent.std,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Saved → {filepath}")


def load_model(agent, filepath):
    if not os.path.exists(filepath):
        print(f"No model found at {filepath}")
        return False
    with open(filepath, 'rb') as f:
        weights = pickle.load(f)
    # actor
    agent.actor_l1.W.data    = weights['actor_l1_W']
    agent.actor_l1.b.data    = weights['actor_l1_b']
    agent.actor_norm1.gamma.data = weights['actor_norm1_g']
    agent.actor_norm1.beta.data  = weights['actor_norm1_b']
    agent.actor_l2.W.data    = weights['actor_l2_W']
    agent.actor_l2.b.data    = weights['actor_l2_b']
    agent.actor_norm2.gamma.data = weights['actor_norm2_g']
    agent.actor_norm2.beta.data  = weights['actor_norm2_b']
    agent.actor_out.W.data   = weights['actor_out_W']
    agent.actor_out.b.data   = weights['actor_out_b']
    # critic
    agent.critic_l1.W.data    = weights['critic_l1_W']
    agent.critic_l1.b.data    = weights['critic_l1_b']
    agent.critic_norm1.gamma.data = weights['critic_norm1_g']
    agent.critic_norm1.beta.data  = weights['critic_norm1_b']
    agent.critic_l2.W.data    = weights['critic_l2_W']
    agent.critic_l2.b.data    = weights['critic_l2_b']
    agent.critic_norm2.gamma.data = weights['critic_norm2_g']
    agent.critic_norm2.beta.data  = weights['critic_norm2_b']
    agent.critic_out.W.data   = weights['critic_out_W']
    agent.critic_out.b.data   = weights['critic_out_b']
    # training state
    agent.std = weights['std']
    print(f"Loaded ← {filepath}")
    return True


def save_log(log_data, filepath):
    import csv
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)