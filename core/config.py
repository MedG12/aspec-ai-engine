from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ASPEC AI Engine"
    
    # Database Settings
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_SERVER: str
    MYSQL_PORT: str
    MYSQL_DB: str
    
    # ChromaDB Settings
    CHROMA_DB_PATH: str
    CHROMA_COLLECTION_NAME: str
    
    # Groq Settings
    GROQ_API_KEY: str
    GROQ_MODEL: str
    
    @property
    def DATABASE_URL(self) -> str:
        # Returns SQLAlchemy connection string for MySQL using PyMySQL
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
        
    class Config:
        env_file = ".env"

settings = Settings()
