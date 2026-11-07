from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_dataset(df):
    return df.drop_duplicates().reset_index(drop=True)

def split_dataset(df,target='Class',test_size=.2,random_state=42):
    X=df.drop(columns=[target]); y=df[target].astype(str)
    return train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)

def scaled_pipeline(classifier):
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('classifier',classifier)])
def unscaled_pipeline(classifier):
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('classifier',classifier)])
