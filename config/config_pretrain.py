config = {
    "data": "nike",

    "wandb_project_name": "",
    "log_wandb" : False,

    "batch_size" : 32,

    # Printing and Logging settings
    "checkpoint_save" : 200,
    "checkpoint_print": 30,

    # Loader
    "mask_ratio" : 0.7,
    "ratio_patches" : 10,
    "clip_grad": 10,
    "warmup_ratio": 0.15,
    "ipe_scale": 1.25,

    #optim
    "lr": 1e-5,
    "end_lr": 1e-4,

    "num_epochs": 201,
    "ema_momentum" : 0.998,

    # Encoder
    "encoder_embed_dim" : 128,
    "encoder_nhead" : 2,
    "encoder_num_layers": 1,
    "encoder_kernel_size" : 3,
    "encoder_embed_bias": True,

    # Predictor
    "predictor_embed" : 128,
    "predictor_nhead" : 2,
    "predictor_num_layers": 1
    }
