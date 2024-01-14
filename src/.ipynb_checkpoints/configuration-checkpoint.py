import json
import cupy as np
import os
from pathlib import Path


class Config:
    hyperParameter = {
        'zero_division_padding': 1e-7,
        'learning_rate': .5e-1,
        'optimizer': {
            'momentum': {
                'initial_momentum': lambda x, y: np.ones(shape=(x, y)),
                'beta': 9e-1,
                'learning_rate_decay': 0.9999
            },

            'sgd': {
                'learning_rate_decay': 0.9999
            },

            'adaGrad': {
                'initial_gradient': lambda x, y: np.zeros(shape=(x, y)),
                'learning_rate_decay': 0.9999
            },

            'rmsProp': {
                'initial_gradient': lambda x, y: np.zeros(shape=(x, y)),
                'beta': 2e-1,
            },

            'adam': {
                'initial_gradient': lambda x, y: np.zeros(shape=(x, y)),
                'initial_momentum': lambda x, y: np.zeros(shape=(x, y)),
                'momentum_beta': 2e-10,
                'gradient_beta': 4e-3,
            }
        },
        'batch_norm': {
            'shift': 1e-1,
            'scale': 5e-1,
        },
        'activation': {
            'leaky_relu': {
                'scale': 1e-3
            },
            'elu': {
                'scale': 1e-3
            },
        },
        'regularizer': {
            'l1': {
                'beta': -2e-2,
            },
            'l2': {
                'beta': -4e-2,
            },
        },
        'weights': {
            'clip': {
                'scale': 1e-1,
                'thr': 4,
            },
            'l2': {
                'beta': 4e-1,
            },
        }

    }


def validateOptions(options: list, option):
    if option not in [option_ for option_ in options]:
        raise ValueError(f'Invalid option! Available options: {", ".join(options)}')


configuration = Config()

if __name__ == '__main__':
    path = Path(os.curdir).absolute()
    print(f'current directory: {path.name}')
    while path.name != 'dota-oracle':
        print(f'searching...')
        path = path.parent

    print(f'setting path: {path.name}')

    base_path = path
    base_model_storage_path = os.path.join(path, 'training_logs')
    try:
        print(f'Connecting to database')
        with open(os.path.join(base_path, 'dbconfig.json'), 'r') as f:
            db_path = json.load(f)
        db_path = os.path.join(db_path['path'])

    except Exception:
        print(f'Failed to read database. Check dbconfig.json')
