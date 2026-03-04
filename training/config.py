"""Hyperparameters for Phase 4 neural network training."""

BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 10
EVAL_CLAMP = 1500       # centipawns — clamp range for normalising labels
CHUNK_SIZE = 500_000    # positions per .npz chunk file

TRAIN_DATA_DIR = "training/data"
WEIGHTS_PATH = "chess_engine/nn/weights.npz"
CHECKPOINT_PATH = "training/best_checkpoint.pt"
