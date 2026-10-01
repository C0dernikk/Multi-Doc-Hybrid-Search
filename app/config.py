from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Hybrid Search Engine"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_memory_only: bool = True
    dense_model_name: str = "all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rrf_k: int = 60
    
    class Config:
        env_file = ".env"

settings = Settings()
