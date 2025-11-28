from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuration(BaseSettings):
    # General
    API_TITLE: str
    API_HOST: str
    API_PORT: int

    # MLFLow
    MLFLOW_TRACKING_URI: str
    MLFLOW_EXPERIMENT_NAME: str

    model_config = SettingsConfigDict(env_file=".env")

    MODEL_URI: str


config = Configuration()
