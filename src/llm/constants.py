import os
import dotenv 


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(PROJECT_ROOT, "../../.env")

dotenv.load_dotenv(ENV_PATH)

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY", "")
TYPHOON_ENDPOINT = os.getenv("TYPHOON_ENDPOINT", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORGANIZATION_ID = os.getenv("OPENAI_ORGANIZATION_ID", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
