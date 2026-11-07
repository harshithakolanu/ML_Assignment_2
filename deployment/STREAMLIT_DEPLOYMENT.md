# Streamlit Community Cloud
1. Push the executed project, five `.joblib` models, test CSV, reports, and figures to GitHub.
2. Sign in at https://share.streamlit.io with GitHub.
3. Select **Create app**, the repository, `main`, and `app.py`.
4. Deploy, review logs, then upload `data/test_data.csv` and test every model.
5. Verify six metrics, confusion matrix, classification report, predictions, and CSV download.

Troubleshooting: keep paths relative; pin the same scikit-learn version used for serialization; confirm all five model files are committed; rebuild after dependency changes; compare the app with notebook results using the same test CSV.
