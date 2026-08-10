config = {
    "data": "NVDA",
    "input_mode": "timeseries",
    "timestamp_col": "Date",

    "series_split_size": 20,
    "patch_size": 5,
    "pretrain_stride": 5,
    "sampling_mode": "sliding_window",
    "normalization": "train_zscore",
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
    "data_end_date": "2026-01-01",
    "validation_fraction": 0.05,
    "target_feature_index": 0,
    "seed": 42,
    "deterministic": True,

    "batch_size" : 32,

    # Printing and Logging settings
    "checkpoint_save" : 500,
    "checkpoint_print": 30,
    "validation_interval": 10,
    "validation_max_batches": None,

    # Loader
    "mask_strategy": "random",
    "mask_ratio" : 0.7,
    "ratio_patches" : 10,
    "clip_grad": 1,
    "ipe_scale": 1.25,
    "lambda_jepa": 1.0,
    "lambda_mae": 0.5,
    "jepa_loss": "mse",
    "mae_loss": "mse",

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
    "predictor_num_layers": 1,

    # MAE reconstruction decoder
    "decoder_type": "residual_mlp",
    "decoder_hidden_dim": 128,
    "decoder_num_layers": 2,
    "decoder_dropout": 0.1,

    # Structured masking
    "mae_window_patches": 1,
    "jepa_gap_patches": 4,
    "jepa_target_patches": 4,
    "anchor_strategy": "random",
    "fixed_anchor": 0,
    "future_target_patches": 4,
    "causal_num_blocks": 2,
    "causal_block_patches": 2,
    "causal_block_gap_patches": 1,

    # Automatically run downstream forecasting after pretraining.
    # The downstream label is Close[t+h] / Close[t] - 1.
    "run_eval": True,
    "eval_use_best": True,
    "eval_checkpoint_to_use": None,
    "eval_encoder_weights": "ema",
    "eval_forecast_target": "relative_return",
    "eval_num_epochs": 501,
    "eval_results_dir": "./results/NVDA/relative_return/seed_42",

    # Optional operational overrides
    "resume_from": None,
    "max_batches_per_epoch": None,
    "save_final": True,
    "path_suffix": None,
    "compatible_save_name": False,
    "notes": "",

    # Inactive unless input_mode="mnist_rows"
    "mnist_root": "./data/MNIST",
    "mnist_train_samples": 512,
    "mnist_val_samples": 128,
    "download_mnist": False,
}
