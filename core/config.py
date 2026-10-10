from pathlib import Path

from dotenv import load_dotenv

# Load ANTHROPIC_API_KEY and friends from .env if it exists
load_dotenv()

################################################################################
################################## Paths #######################################
################################################################################

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
APP_DATA_DIR = DATA_DIR / "app"

REPORTS_DIR = PROJ_ROOT / "reports"
EVAL_DIR = REPORTS_DIR / "eval"

################################################################################
############################ Global Constants ##################################
################################################################################

rstate = 222  # random state for reproducibility

################################################################################
############################# Source Dataset ###################################
################################################################################

# McAuley Lab, Amazon Reviews'23 (Hou et al., 2024). Academic, non-commercial.
MCAULEY_BASE = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw"
)
META_URL = f"{MCAULEY_BASE}/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl.gz"
REVIEW_URL = f"{MCAULEY_BASE}/review_categories/Grocery_and_Gourmet_Food.jsonl.gz"

DATASET_CITATION = (
    "Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). "
    "Bridging Language and Items for Retrieval and Recommendation. "
    "arXiv:2403.03952."
)
