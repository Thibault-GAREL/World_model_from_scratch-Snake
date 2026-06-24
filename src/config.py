from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Paths
    DATA_RAW: Path = ROOT_DIR / "data" / "1-raw"
    DATA_PROCESSED: Path = ROOT_DIR / "data" / "2-processed"
    DATA_EXTERNAL: Path = ROOT_DIR / "data" / "3-external"
    OUTPUTS_MODELS: Path = ROOT_DIR / "outputs" / "models"
    OUTPUTS_LOGS: Path = ROOT_DIR / "outputs" / "logs"
    OUTPUTS_RESULTS: Path = ROOT_DIR / "outputs" / "results"

    # Reproducibility
    RANDOM_STATE: int = 42

    # MLflow
    MLFLOW_EXPERIMENT_NAME: str = "jepa-world-model"

    # --- Data ---
    TRANSITIONS_FILE: str = "snake_transitions.npz"   # under DATA_PROCESSED
    VAL_FRACTION: float = 0.1

    # --- Model ---
    MODEL_NAME: str = "jepa-world-model"
    EMBED_DIM: int = 128
    HIDDEN_DIM: int = 256

    # --- Training ---
    DEVICE: str = "auto"          # "auto" | "cuda" | "cpu"
    EPOCHS: int = 80
    PATIENCE: int = 15            # early-stopping patience (epochs)
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 1.0
    ROLLOUT_K: int = 5            # multi-step latent unroll length

    # --- JEPA / anti-collapse ---
    EMA_DECAY: float = 0.99       # target encoder momentum
    PRED_COEF: float = 25.0       # latent prediction loss weight
    STD_COEF: float = 25.0        # VICReg variance hinge weight
    COV_COEF: float = 1.0         # VICReg covariance weight
    REWARD_COEF: float = 10.0     # reward head (MSE) weight (sparse signal -> upweighted)
    REWARD_EVENT_WEIGHT: float = 5.0   # extra weight on non-zero rewards (food / death)
    DONE_COEF: float = 2.0        # done head (BCE) weight
    DONE_POS_WEIGHT_CAP: float = 15.0  # cap for BCE pos_weight (class imbalance)

    # --- Planning (MPC) ---
    MPC_HORIZON: int = 5          # planning depth at play time (best by horizon sweep)
    MPC_GAMMA: float = 0.99       # discount in rollout scoring


config = Config()
