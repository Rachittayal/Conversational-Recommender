import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY", "")
GROQ_MODEL="llama-3.3-70b-versatile"

CATALOG_URL=("https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json")
CATALOG_CACHE_PATH = "data/catalog.json"

MAX_TURNS=8
MAX_CLARIFY_QUESTIONS=2

REQUEST_TIMEOUT_SECONDS=25
EMBEDDING_MODEL="all-MiniLM-L6-v2"

TOP_K_RESULTS=20
MAX_RECOMMENDATIONS=10

TEST_TYPE_MAPPING: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B", 
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",  
    "Simulations": "S",
}
CODE_TO_LABEL: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}