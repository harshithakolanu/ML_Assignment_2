from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def save_eda(df,target,figure_dir):
    p=Path(figure_dir); p.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(9,5)); sns.countplot(data=df,x=target,order=df[target].value_counts().index); plt.xticks(rotation=30); plt.tight_layout(); plt.savefig(p/'class_distribution.png',dpi=160); plt.close()
    df.drop(columns=[target]).hist(figsize=(16,12),bins=30); plt.tight_layout(); plt.savefig(p/'feature_distributions.png',dpi=160); plt.close()
    plt.figure(figsize=(14,10)); sns.heatmap(df.drop(columns=[target]).corr(),cmap='coolwarm',center=0); plt.tight_layout(); plt.savefig(p/'correlation_heatmap.png',dpi=160); plt.close()
    plt.figure(figsize=(16,9)); sns.boxplot(data=df.drop(columns=[target]),orient='h'); plt.tight_layout(); plt.savefig(p/'feature_boxplots.png',dpi=160); plt.close()

def save_comparison(results,figure_dir):
    p=Path(figure_dir); p.mkdir(parents=True,exist_ok=True)
    ax=results.set_index('ML Model Name').plot(kind='bar',figsize=(13,6),ylim=(0,1)); ax.set_ylabel('Score'); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(p/'model_comparison.png',dpi=160); plt.close()

def save_confusion(cm,classes,name,figure_dir):
    plt.figure(figsize=(7,6)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout(); plt.savefig(Path(figure_dir)/f'confusion_matrix_{name}.png',dpi=160); plt.close()
