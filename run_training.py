import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from config import *
from src.data_loader import load_from_uci,validate_dataset
from src.preprocessing import clean_dataset,split_dataset
from src.train_models import build_models,train_and_save
from src.evaluate_models import evaluate_all
from src.visualization import save_eda,save_comparison,save_confusion

def main():
    df=load_from_uci(RAW_DATA_PATH); print(validate_dataset(df))
    df=clean_dataset(df); df.to_csv(PROCESSED_DATA_PATH,index=False)
    save_eda(df,TARGET_COLUMN,FIGURE_DIR)
    Xtr,Xte,ytr,yte=split_dataset(df,TARGET_COLUMN,TEST_SIZE,RANDOM_STATE)
    test=Xte.copy(); test[TARGET_COLUMN]=yte.values; test.to_csv(TEST_DATA_PATH,index=False)
    models=train_and_save(build_models(RANDOM_STATE),Xtr,ytr,MODEL_DIR)
    results,details=evaluate_all(models,Xte,yte,REPORT_DIR); save_comparison(results,FIGURE_DIR)
    for n,m in models.items(): save_confusion(details[n][2],m.classes_,n,FIGURE_DIR)
    print(results.to_string(index=False))
if __name__=='__main__': main()
