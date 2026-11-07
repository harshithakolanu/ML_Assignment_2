from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from .preprocessing import scaled_pipeline, unscaled_pipeline

def build_models(random_state=42):
    return {
      'logistic_regression': scaled_pipeline(LogisticRegression(max_iter=3000,random_state=random_state)),
      'decision_tree': unscaled_pipeline(DecisionTreeClassifier(max_depth=14,min_samples_leaf=2,random_state=random_state)),
      'knn': scaled_pipeline(KNeighborsClassifier(n_neighbors=7,weights='distance')),
      'naive_bayes': scaled_pipeline(GaussianNB()),
      'svm': scaled_pipeline(SVC(C=3.0,kernel='rbf',probability=True,random_state=random_state))}

def train_and_save(models,X_train,y_train,model_dir):
    Path(model_dir).mkdir(parents=True,exist_ok=True)
    for name,model in models.items(): model.fit(X_train,y_train); joblib.dump(model,Path(model_dir)/f'{name}.joblib')
    return models
