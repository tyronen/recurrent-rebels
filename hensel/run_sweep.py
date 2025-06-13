import wandb
from sweep_config import init_sweep
from train import train

if __name__ == '__main__':
    # Initialize the sweep
    sweep_id = init_sweep()
    
    # Run the sweep
    wandb.agent(sweep_id, function=train, count=20)  # Run 20 trials 