# GitHub deployment
1. Create an empty GitHub repository.
2. From this project directory run:
```bash
git init
git branch -M main
git add .
git commit -m "Initial project structure and Dry Bean dataset loader"
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```
Use genuine later commits such as `Add EDA and leakage-safe preprocessing`, `Train and evaluate five classifiers`, `Add saved pipelines and Streamlit app`, and `Finalize report and deployment documentation`.
Never commit `kaggle.json` or API tokens.
