config = {
    "data": "NVDA",

    "pretrain_until_index": 999,
    "series_split_size": 60,
    "patch_size": 5,    
    "normalize_on_train_only": True,
    "feature_cols": [
        "Close",
        "Volume",
        "MA10",
        "MA50",
        "sentiment_mean",
    ],
    "sentiment_path": "./NVDA_daily_sentiment.csv",
    "train_end_date": "2024-12-31",
    "test_start_date": "2025-01-01",
    "validation_fraction": 0.05,
    "test_fraction": 0.15,
    "target_feature_index": 0,

    "wandb_project_name": "",
    "log_wandb" : False,

    "batch_size" : 32,

    # Printing and Logging settings
    "checkpoint_save" : 500,
    "checkpoint_print": 30,

    # Loader
    "mask_ratio" : 0.7,
    "ratio_patches" : 10,
    "clip_grad": 1,
    "warmup_ratio": 0.15,
    "ipe_scale": 1.25,
    "lr_pretrain": 1e-05,
    "pooling": "Mean",

    #optim
    "lr": 1e-5,
    "end_lr": 1e-6,

    "num_epochs": 2001,
    "ema_momentum" : 0.998,

    # Encoder
    "encoder_embed_dim" : 256,
    "encoder_nhead" : 2,
    "encoder_num_layers": 1,
    "encoder_kernel_size" : 3,
    "encoder_embed_bias": True,

    # Predictor
    "predictor_embed" : 128,
    "predictor_nhead" : 2,
    "predictor_num_layers": 1
    }
