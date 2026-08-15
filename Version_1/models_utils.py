'''
Handles saving and loading model weights with full backwards-compatibility.
'''

import pickle
import os
import csv
import numpy as np


def _extractor_weights(agent):
    weights = {}
    for name, module in [
        ('lstm',      agent.lstm),
        ('attention', agent.attention),
        ('cnn',       agent.cnn),
        ('regime',    agent.regime),
        ('fusion',    agent.fusion),
    ]:
        if module is None:
            continue
        for i, p in enumerate(module.parameters()):
            weights[f'{name}_p{i}'] = p.data.copy()
    return weights


def save_model(agent, filepath):
    # sanity check before saving
    for name, module in [('lstm', agent.lstm), ('cnn', agent.cnn),
                         ('fusion', agent.fusion), ('regime', agent.regime)]:
        if module is None:
            continue
        for p in module.parameters():
            if not np.isfinite(p.data).all():
                print(f"  [WARNING] NaN/inf detected in {name} — skipping save to protect checkpoint")
                return False

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    weights = {
        # actor
        'actor_l1_W':    agent.actor_l1.W.data.copy(),
        'actor_l1_b':    agent.actor_l1.b.data.copy(),
        'actor_norm1_g': agent.actor_norm1.gamma.data.copy(),
        'actor_norm1_b': agent.actor_norm1.beta.data.copy(),
        'actor_l2_W':    agent.actor_l2.W.data.copy(),
        'actor_l2_b':    agent.actor_l2.b.data.copy(),
        'actor_norm2_g': agent.actor_norm2.gamma.data.copy(),
        'actor_norm2_b': agent.actor_norm2.beta.data.copy(),
        
        # Split Policy Heads
        'actor_dir_W':   agent.actor_dir.W.data.copy(),
        'actor_dir_b':   agent.actor_dir.b.data.copy(),
        'actor_size_W':  agent.actor_size.W.data.copy(),
        'actor_size_b':  agent.actor_size.b.data.copy(),
        
        'critic_l1_W':    agent.critic_l1.W.data.copy(),
        'critic_l1_b':    agent.critic_l1.b.data.copy(),
        'critic_norm1_g': agent.critic_norm1.gamma.data.copy(),
        'critic_norm1_b': agent.critic_norm1.beta.data.copy(),
        'critic_l2_W':    agent.critic_l2.W.data.copy(),
        'critic_l2_b':    agent.critic_l2.b.data.copy(),
        'critic_norm2_g': agent.critic_norm2.gamma.data.copy(),
        'critic_norm2_b': agent.critic_norm2.beta.data.copy(),
        
        # Unified Critic Output + Duplicate reference maps for seamless legacy loading checks
        'critic_out_W':     agent.critic_out.W.data.copy(),
        'critic_out_b':     agent.critic_out.b.data.copy(),
        'critic_long_W':    agent.critic_out.W.data.copy(),
        'critic_long_b':    agent.critic_out.b.data.copy(),
        'critic_short_W':   agent.critic_out.W.data.copy(),
        'critic_short_b':   agent.critic_out.b.data.copy(),
        'critic_neutral_W': agent.critic_out.W.data.copy(),
        'critic_neutral_b': agent.critic_out.b.data.copy(),
        
        **_extractor_weights(agent),
        'std': agent.std,
    }
    with open(filepath, 'wb') as f:
        pickle.dump(weights, f)
    print(f"Saved → {filepath}")


def load_model(agent, filepath):
    if not os.path.exists(filepath):
        print(f"No model found at {filepath}")
        return False

    with open(filepath, 'rb') as f:
        weights = pickle.load(f)

    def safe_load(layer, name_w, name_b):
        if name_w in weights and name_b in weights:
            layer.W.data = weights[name_w]
            layer.b.data = weights[name_b]
        else:
            print(f"  [LOAD INFO] Layer {name_w}/{name_b} not found in checkpoint; skipped.")

    def safe_load_norm(norm, name_g, name_b):
        if name_g in weights and name_b in weights:
            norm.gamma.data = weights[name_g]
            norm.beta.data = weights[name_b]

    safe_load(agent.actor_l1, 'actor_l1_W', 'actor_l1_b')
    safe_load_norm(agent.actor_norm1, 'actor_norm1_g', 'actor_norm1_b')
    safe_load(agent.actor_l2, 'actor_l2_W', 'actor_l2_b')
    safe_load_norm(agent.actor_norm2, 'actor_norm2_g', 'actor_norm2_b')
    
    # Split Actor Head loading with backwards mapping bootstrap from old actor_out
    if 'actor_out_W' in weights and 'actor_out_b' in weights:
        agent.actor_size.W.data = weights['actor_out_W'][:, 1:2].copy()
        agent.actor_size.b.data = weights['actor_out_b'][1:2].copy()
        print("  [LOAD INFO] Bootstrapped actor_size from actor_out column 1.")
    else:
        safe_load(agent.actor_dir, 'actor_dir_W', 'actor_dir_b')
        safe_load(agent.actor_size, 'actor_size_W', 'actor_size_b')

    safe_load(agent.critic_l1, 'critic_l1_W', 'critic_l1_b')
    safe_load_norm(agent.critic_norm1, 'critic_norm1_g', 'critic_norm1_b')
    safe_load(agent.critic_l2, 'critic_l2_W', 'critic_l2_b')
    safe_load_norm(agent.critic_norm2, 'critic_norm2_g', 'critic_norm2_b')
    
    # Dynamic loader compatibility for Unified Critic Head mapping smoothly
    if 'critic_out_W' in weights and 'critic_out_b' in weights:
        agent.critic_out.W.data = weights['critic_out_W'].copy()
        agent.critic_out.b.data = weights['critic_out_b'].copy()
        print("  [LOAD INFO] Loaded unified critic head.")
    elif 'critic_long_W' in weights and 'critic_long_b' in weights:
        agent.critic_out.W.data = weights['critic_long_W'].copy()
        agent.critic_out.b.data = weights['critic_long_b'].copy()
        print("  [LOAD INFO] Bootstrapped unified critic head from legacy triple value weights.")

    for name, module in [
        ('lstm',      agent.lstm),
        ('attention', agent.attention),
        ('cnn',       agent.cnn),
        ('regime',    agent.regime),
        ('fusion',    agent.fusion),
    ]:
        if module is None:
            continue
        for i, p in enumerate(module.parameters()):
            key = f'{name}_p{i}'
            if key in weights:
                p.data = weights[key]
                
    if 'std' in weights:
        agent.std = weights['std']

    print(f"Loaded ← {filepath}")
    return True


def save_log(log_data: dict, filepath: str):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    file_exists = os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(log_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)