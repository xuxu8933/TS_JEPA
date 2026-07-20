config = {
    # Data Path
    "data" : "NVDA",
    "path_save" : "./logs/output_model/",
    "results_dir" : "./results",

    # Unified checkpoint selection used by eval_dual_loss.py:
    #   best  -> the matching *_best.pt checkpoint
    #   last  -> the matching checkpoint with the largest epoch number
    #   epoch -> checkpoint_to_use
    #   path  -> pretrain_checkpoint_path
    "checkpoint_selection": "last",
    "pretrain_checkpoint_path": None,
    "mask_strategy": "random",
    "lambda_jepa": 1.0,
    "lambda_mae": 1.0,
    "mae_window_patches": 1,
    "jepa_gap_patches": 4,
    "jepa_target_patches": 4,
    "future_target_patches": 4,
    "causal_num_blocks": 2,
    "causal_block_patches": 2,
    "causal_block_gap_patches": 1,

    "sampling_mode": "sliding_window",
    "normalization": "window_return",
    "normalization_stats": None,
    "pretrain_encoder_weights": "ema",

    "target_feature_index": 0,
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

    # Loader
    "ratio_patches" : 10,
    "ratio_supervision": 1.0,

    # Optim
    "num_epochs": 501,
    "batch_size" : 32,
    "lr": 1e-03,
    "decoder_type": "residual_mlp",
    "decoder_hidden_dim": 128,
    "decoder_num_layers": 2,
    "decoder_dropout": 0.1,
    "fine_tune_encoder": True,
    "encoder_finetune_lr": 1e-5,
    "trend_weight": 0.001,
    "trend_loss_temperature": 0.01,
    "trend_loss_threshold": 1e-5,
    "trend_selection_weight": 0.0005,

    # CNN Model
    "cnn_out_channels": [32, 64, 128],
    "cnn_kernel_size" : 3,
    "cnn_dense_dim" : 32,

    # Transformer Model
    "embed_dim" : 128,
    "nhead" : 2,
    "num_layers": 1,
    "kernel_size" : 3,
    "transformer_dense_dim": 64,
    "pooling": "Mean",

    # Pretrained Transformer -- Config
    "pretrain_encoder_embed_dim" : 256,
    "pretrain_encoder_nhead" : 2,
    "pretrain_encoder_num_layers": 1,
    "pretrain_encoder_kernel_size" : 3,
    "pretrain_encoder_embed_bias" : True,
    "pretrain_transformer_dense_dim" : 128,

    "pretrain_decoder_embed_dim" : 128,
    "pretrain_decoder_nhead" : 2,
    "pretrain_decoder_num_layers": 1,

    "checkpoint_to_use": 2000,
    "lr_pretrain": 1e-05,
    "mask_ratio" : 0.7,
    "ema_pretrain" : 0.998,

    "eval_type" : "last" # or "last"
}
