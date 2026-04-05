"""
Handles saving and loading model weights using torch.save/load.
Same interface as before — save_model(agent, path) / load_model(agent, path).
"""

import os
import csv
import torch


def save_model(agent, filepath):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    checkpoint = {
        'actor':agent.actor.state_dict(),
        'critic':agent.critic.state_dict(),
        'std':agent.std,
        'return_rms': {
            'mean':agent.return_rms.mean,
            'var':agent.return_rms.var,
            'count': agent.return_rms.count,
        },
    }

    # Save extractor weights if registered
    for name, module in [('lstm',  agent.lstm),
                         ('attention', agent.attention),
                         ('cnn',agent.cnn),
                         ('regime', agent.regime),
                         ('fusion',agent.fusion)]:
        if module is not None:
            checkpoint[f'extractor_{name}'] = module.state_dict()

    torch.save(checkpoint, filepath)
    print(f"Saved → {filepath}")


def load_model(agent, filepath):
    if not os.path.exists(filepath):
        print(f"No model found at {filepath}")
        return False

    checkpoint = torch.load(filepath, map_location='cpu')

    agent.actor.load_state_dict(checkpoint['actor'])
    agent.critic.load_state_dict(checkpoint['critic'])
    agent.std = checkpoint['std']

    if 'return_rms' in checkpoint:
        rms = checkpoint['return_rms']
        agent.return_rms.mean  = rms['mean']
        agent.return_rms.var   = rms['var']
        agent.return_rms.count = rms['count']

    # Restore extractor weights if present
    for name, module in [('lstm',      agent.lstm),
                         ('attention', agent.attention),
                         ('cnn',       agent.cnn),
                         ('regime',    agent.regime),
                         ('fusion',    agent.fusion)]:
        key = f'extractor_{name}'
        if key in checkpoint and module is not None:
            module.load_state_dict(checkpoint[key])

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