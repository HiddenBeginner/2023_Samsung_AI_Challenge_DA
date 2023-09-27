config = {
    'dir_data': './data',
    'dir_ckpt': './ckpt',
    'seed': 0,
    'n_epochs': 50,
    'optimizer': {
        'lr': 1e-5,
    },

    'scheduler': {
        'T_0': 50,
        'T_mult': 1,
        'eta_max': 0.0001,
        'T_up': 5,
        'gamma': 0.5,
    },

    'optimizer_D': {
        'lr': 1e-3,
    },
    'scheduler_D': {
        'T_0': 50,
        'T_mult': 1,
        'eta_max': 0.0001,
        'T_up': 5,
        'gamma': 0.5,
    },

    'wandb': {
        'project': 'DA',
        'name': 'SegFormer-b5_512_MultiScale_Full_FocalLoss'
    }
}


pseudo_labeling_config = {
    'dir_data': './data',
    'dir_ckpt': './ckpt',
    'seed': 0,
    'n_epochs': 4,
    'optimizer': {
        'lr': 0.0,
    },

    'scheduler': {
        'T_0': 4,
        'T_mult': 1,
        'eta_max': 0.00001,
        'T_up': 4,
        'gamma': 0.5,
    },

    'wandb': {
        'project': 'DA',
        'name': 'SegFormer-b5_512_PseudoLabeling'
    }
}
