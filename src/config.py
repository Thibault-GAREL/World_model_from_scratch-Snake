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

    # --- Environment ---
    ENV_MAX_STEPS: int = 1000     # episode cap (500 mechanically caps the score ~30)
    R_SHAPING: float = 0.4        # potential-based guidance per cell closer to the apple
    #   Scale matters for *learnability*, not for optimality: potential-based
    #   shaping leaves the optimal policy unchanged at any scale, but a +-0.1
    #   target next to +-1 food/death events contributes ~1% of the reward MSE,
    #   so the head simply never learns it (measured MAE 0.276 at 0.1).

    # --- Data ---
    TRANSITIONS_FILE: str = "snake_transitions.npz"   # under DATA_PROCESSED
    VAL_FRACTION: float = 0.1
    DATA_N_TRANSITIONS: int = 200_000
    DATA_EPSILON: float = 0.25    # fully random actions (they kill -> done signal)
    DATA_SAFE_EXPLORE: float = 0.25    # random *legal* actions (action coverage, no death)
    DATA_CLEAN_FRACTION: float = 0.6   # share of TRANSITIONS from the pure oracle
    DATA_ORACLE_SAFE: bool = True  # flood-fill safety check (reaches long snakes)
    DATA_ON_DEVICE: bool = True   # keep the dataset in VRAM (kills the per-batch copy)
    DATA_DEVICE_BUDGET_GB: float = 2.5   # above this, stream from RAM instead

    # --- Model ---
    MODEL_NAME: str = "jepa-world-model"
    EMBED_DIM: int = 256          # best of the RunPod sweep (128 scored lower)
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
    # RunPod sweep 2026-08-05 (8 configs, 200k transitions, 40 epochs):
    # REWARD_COEF is the dominant knob by far. The four runs at 50 took the four
    # top places, the four at 10 the bottom four. Raising it lifted MPC from 3.75
    # to 10.65, action-ranking from 38% to 47% and safe-action rate from 88% to
    # 94%: the reward head was simply drowned by the latent prediction loss.
    PRED_COEF: float = 5.0        # latent prediction loss weight (25 scored lower)
    STD_COEF: float = 25.0        # VICReg variance hinge weight
    COV_COEF: float = 1.0         # VICReg covariance weight
    REWARD_COEF: float = 50.0     # reward head (MSE) weight (10 scored far worse)
    REWARD_EVENT_WEIGHT: float = 5.0   # extra weight on non-zero rewards (food / death)
    DONE_COEF: float = 2.0        # done head (BCE) weight
    DONE_POS_WEIGHT_CAP: float = 40.0  # cap for BCE pos_weight (class imbalance)
    SPACE_COEF: float = 2.0       # free-space head (BCE) weight, 0 disables the head

    # --- Planning (MPC) ---
    MPC_HORIZON: int = 5          # planning depth at play time (best by horizon sweep)
    MPC_GAMMA: float = 0.99       # discount in rollout scoring
    MPC_SPACE_COEF: float = 0.3   # weight of the predicted free space in the score

    # --- DAgger-style iterative data collection ---
    DAGGER_ROUNDS: int = 3        # collect-with-planner -> retrain cycles
    DAGGER_TRANSITIONS: int = 100_000   # new transitions collected per round
    DAGGER_EPSILON: float = 0.05  # exploration noise while collecting with the planner
    DAGGER_KEEP: float = 0.6      # fraction of the previous dataset kept each round

    # --- Evaluation ---
    EVAL_EPISODES: int = 30
    EVAL_MAX_STEPS: int = 1000


config = Config()
