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
    MLFLOW_EXPERIMENT_NAME: str = "default"


config = Config()
