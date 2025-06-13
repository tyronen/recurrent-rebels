import wandb

sweep_config = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': -8,  # 1e-8
            'max': -3,  # 1e-3
        },
        'weight_decay': {
            'distribution': 'log_uniform',
            'min': -8,  # 1e-8
            'max': -3,  # 1e-3
        },
        'hidden_size': {
            'values': [128, 256, 512]
        },
        'dropout_rate': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },
        'num_epochs': {
            'values': [5, 10, 15]
        }
    }
}

def init_sweep():
    sweep_id = wandb.sweep(sweep_config, project="hensel-model")
    return sweep_id 