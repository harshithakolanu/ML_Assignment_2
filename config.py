from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "dry_bean.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "dry_bean_clean.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.20
UCI_ID = 602
EXPECTED_FEATURES = ["Area","Perimeter","MajorAxisLength","MinorAxisLength","AspectRation","Eccentricity","ConvexArea","EquivDiameter","Extent","Solidity","roundness","Compactness","ShapeFactor1","ShapeFactor2","ShapeFactor3","ShapeFactor4"]
