import os
from dotenv import load_dotenv

load_dotenv()

def int_env(name: str, default: int):
    try:
        return int(os.getenv(name, default))
    except:
        return default
