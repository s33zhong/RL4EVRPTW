from lightning.pytorch.callbacks import Callback
import pandas as pd
import numpy as np
from datetime import datetime
import torch

def validity_check(rewards, raw=False):
    valid_rewards = []
    for reward in rewards:
        if -reward > 1000:
            pass
        else:
            valid_rewards.append(reward)
    return np.array(valid_rewards)

def get_reward_and_check(policy, test_data, env_scale):
    rewards_trained = []
    rewards_trained_for_fesibility = []
    num_valids = []
    for td_i, env_i in zip(test_data, env_scale):
        out = policy(td_i.clone(), 
                    env=env_i, 
                    phase="test", 
                    feasibility_check=True, 
                    decode_type="greedy", 
                    return_actions=True)
        valid_out = validity_check(out['reward'].cpu().numpy())
        rewards_trained.append(valid_out)
        num_valids.append(len(valid_out))

    return rewards_trained, num_valids

class RewardLoggingCallback(Callback):
    def __init__(self, policy, test_data, env_scale, scale, log_dir="logs", file_name=None):
        super().__init__()
        self.policy = policy
        self.test_data = test_data
        self.env_scale = env_scale
        self.scale = scale

        # Generate log file path with current date-time
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"{log_dir}/{file_name}_{current_time}.xlsx"
        self.logs = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Switch to evaluation mode
        self.policy.eval()
        with torch.no_grad():  # Disable gradient computation
            rewards_trained, num_valid = get_reward_and_check(self.policy, self.test_data, self.env_scale)
        
        # Log and save results
        epoch_data = {"epoch": trainer.current_epoch + 1}
        for i, s in enumerate(self.scale):
            epoch_data[f"C{s}_mean_reward"] = -rewards_trained[i].mean()
            epoch_data[f"C{s}_feasible_counts"] = num_valid[i]
        self.logs.append(epoch_data)

        # Save logs
        df = pd.DataFrame(self.logs)
        df.to_excel(self.log_path, index=False)
        
        # Restore to training mode
        self.policy.train()
        
        # print('Callback is finished')
