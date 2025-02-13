import os
from dotenv import load_dotenv

# .envファイルの内容を読み込む
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "./model")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
DATABASE_PATH = os.getenv("DATABASE_PATH", "./src/registered_faces.json")
