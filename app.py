from pathlib import Path
import streamlit as st, pandas as pd, joblib, matplotlib.pyplot as plt, seaborn as sns
from src.evaluate_models import evaluate_model
from config import EXPECTED_FEATURES,TARGET_COLUMN,MODEL_DIR
st.set_page_config(page_title='Dry Bean Model Lab',page_icon='🫘',layout='wide')
st.title('🫘 Dry Bean Classification Model Lab')
st.caption('M.Tech AIML Assignment 2 | Saved-pipeline evaluation only | No retraining')
st.info('Dataset: UCI Dry Bean, 13,611 instances, 16 numerical features, 7 classes.')
models={'Logistic Regression':'logistic_regression.joblib','Decision Tree':'decision_tree.joblib','K-Nearest Neighbors':'knn.joblib','Gaussian Naive Bayes':'naive_bayes.joblib','Support Vector Machine':'svm.joblib'}
upload=st.file_uploader('Upload held-out test CSV containing all features and Class',type='csv')
mode=st.radio('Evaluation mode',['Selected model','Compare all five'],horizontal=True)
choice=st.selectbox('Model',list(models))
if upload:
 try:
  df=pd.read_csv(upload)
  if df.empty: raise ValueError('The uploaded file is empty.')
  missing=[c for c in EXPECTED_FEATURES+[TARGET_COLUMN] if c not in df.columns]
  extra=[c for c in df.columns if c not in EXPECTED_FEATURES+[TARGET_COLUMN]]
  if missing: raise ValueError(f'Missing columns: {missing}')
  if extra: raise ValueError(f'Unexpected columns: {extra}')
  X=df[EXPECTED_FEATURES].apply(pd.to_numeric,errors='raise'); y=df[TARGET_COLUMN].astype(str)
  selected=models if mode.startswith('Compare') else {choice:models[choice]}
  rows=[]; last=None
  for label,file in selected.items():
   path=MODEL_DIR/file
   if not path.exists(): raise FileNotFoundError(f'{path} not found. Run python run_training.py first.')
   model=joblib.load(path); met,pred,report,cm=evaluate_model(label,model,X,y); rows.append(met); last=(label,model,pred,report,cm)
  res=pd.DataFrame(rows); st.subheader('Evaluation metrics'); st.dataframe(res.style.format({c:'{:.4f}' for c in ['Accuracy','AUC','Precision','Recall','F1','MCC']}),width="stretch")
  st.bar_chart(res.set_index('ML Model Name'))
  label,model,pred,report,cm=last; c1,c2=st.columns(2)
  with c1:
   st.subheader(f'Confusion matrix: {label}'); fig,ax=plt.subplots(); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=model.classes_,yticklabels=model.classes_,ax=ax); ax.set(xlabel='Predicted',ylabel='Actual'); st.pyplot(fig)
  with c2: st.subheader('Classification report'); st.dataframe(pd.DataFrame(report).T,width="stretch")
  out=df.copy(); out['Predicted_Class']=pred; st.subheader('Predictions'); st.dataframe(out[[TARGET_COLUMN,'Predicted_Class']].head(100)); st.download_button('Download predictions',out.to_csv(index=False).encode(),'dry_bean_predictions.csv','text/csv')
 except Exception as e: st.error(str(e))
