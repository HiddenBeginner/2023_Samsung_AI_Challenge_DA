config = {
    'dir_data': './data',
    'dir_ckpt': './ckpt',
    'seed': 0,
    'n_epochs': 50,
    'optimizer': {
        'lr': 0.0,
    },

    'criterion': {
        'dice': False
    },

    'scheduler': {
        'T_0': 50,
        'T_mult': 1,
        'eta_max': 0.0001,
        'T_up': 5,
        'gamma': 0.5,
    },

    'wandb': {
        'project': 'DA',
        'name': 'SegFormer-b5'
    }
}


pseudo_labeling_config = {
    'dir_data': './data',
    'dir_ckpt': './ckpt',
    'seed': 0,
    'n_epochs': 5,
    'optimizer': {
        'lr': 0.0,
    },

    'scheduler': {
        'T_0': 5,
        'T_mult': 1,
        'eta_max': 0.00001,
        'T_up': 5,
        'gamma': 0.5,
    },

    'criterion': {
        'dice': False
    },

    'wandb': {
        'project': 'DA',
        'name': 'SegFormer-b5_PseudoLabeling'
    }
}
