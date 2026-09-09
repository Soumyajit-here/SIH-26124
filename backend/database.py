import os
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build the database URL from individual environment variables, with a fallback
DB_USER = os.getenv("DB_USER", "root")
raw_password = os.getenv("DB_PASSWORD", "root")
DB_PASSWORD = urllib.parse.quote_plus(raw_password)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "sih26124")

from sqlalchemy.pool import StaticPool

# Check if we are running in a test environment
IS_TESTING = os.getenv("TESTING") == "1"

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

if IS_TESTING:
    DATABASE_URL = "sqlite:///:memory:"
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
else:
    # Using pymysql as the driver
    DATABASE_URL = os.getenv("DATABASE_URL", f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a declarative base class
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
