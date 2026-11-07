from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

def normalize_schema(df: pd.DataFrame, target_column="Class") -> pd.DataFrame:
    df=df.copy(); df.columns=[str(c).strip() for c in df.columns]
    aliases={"AspectRatio":"AspectRation","Roundness":"roundness"}
    df=df.rename(columns=aliases)
    if target_column not in df.columns: raise ValueError(f"Target column '{target_column}' not found. Found: {list(df.columns)}")
    if df.shape[1] != 17: raise ValueError(f"Expected 16 features plus target; found {df.shape[1]} columns")
    return df

def load_from_uci(save_path=None):
    bean=fetch_ucirepo(id=602)
    X=bean.data.features.copy(); y=bean.data.targets.copy()
    y.columns=["Class"]
    df=normalize_schema(pd.concat([X,y],axis=1))
    if save_path:
        Path(save_path).parent.mkdir(parents=True,exist_ok=True); df.to_csv(save_path,index=False)
    return df

def load_from_kaggle_file(path, target_column="Class"):
    path=Path(path)
    if path.suffix.lower()=='.csv': df=pd.read_csv(path)
    elif path.suffix.lower() in {'.xlsx','.xls'}: df=pd.read_excel(path,engine='openpyxl')
    else: raise ValueError('Use a CSV or XLSX file.')
    if target_column!='Class': df=df.rename(columns={target_column:'Class'})
    return normalize_schema(df)

def validate_dataset(df):
    return {"rows":len(df),"columns":df.shape[1],"features":df.shape[1]-1,"target":"Class","missing":int(df.isna().sum().sum()),"duplicates":int(df.duplicated().sum()),"classes":df['Class'].value_counts().to_dict(),"dtypes":df.dtypes.astype(str).to_dict()}
