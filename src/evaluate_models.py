from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef,classification_report,confusion_matrix
from sklearn.preprocessing import label_binarize

def evaluate_model(name,model,X,y):
    pred=model.predict(X); classes=model.classes_
    score=model.predict_proba(X) if hasattr(model,'predict_proba') else model.decision_function(X)
    y_bin=label_binarize(y,classes=classes)
    auc=roc_auc_score(y_bin,score,average='macro',multi_class='ovr')
    metrics={'ML Model Name':name,'Accuracy':accuracy_score(y,pred),'AUC':auc,'Precision':precision_score(y,pred,average='macro',zero_division=0),'Recall':recall_score(y,pred,average='macro',zero_division=0),'F1':f1_score(y,pred,average='macro',zero_division=0),'MCC':matthews_corrcoef(y,pred)}
    return metrics,pred,classification_report(y,pred,output_dict=True,zero_division=0),confusion_matrix(y,pred,labels=classes)

def evaluate_all(models,X,y,report_dir):
    report_dir=Path(report_dir); (report_dir/'classification_reports').mkdir(parents=True,exist_ok=True)
    rows=[]; details={}
    for name,m in models.items():
        met,pred,rep,cm=evaluate_model(name,m,X,y); rows.append(met); details[name]=(pred,rep,cm)
        pd.DataFrame(rep).T.to_csv(report_dir/'classification_reports'/f'{name}.csv')
    out=pd.DataFrame(rows).sort_values(['F1','MCC','AUC'],ascending=False); out.to_csv(report_dir/'model_comparison.csv',index=False)
    return out,details
