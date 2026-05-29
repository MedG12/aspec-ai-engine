from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ASPEC AI Engine"
    
    # Database Settings
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = ""
    MYSQL_SERVER: str = "localhost"
    MYSQL_PORT: str = "3306"
    MYSQL_DB: str = "aspec_db"
    
    # ChromaDB Settings
    CHROMA_DB_PATH: str = "chroma_db"
    CHROMA_COLLECTION_NAME: str = "aspec_full_context_v5"
    
    @property
    def DATABASE_URL(self) -> str:
        # Returns SQLAlchemy connection string for MySQL using PyMySQL
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
        
    class Config:
        env_file = ".env"

settings = Settings()
