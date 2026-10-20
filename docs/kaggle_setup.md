# Kaggle: setting up the seven-class EDA

1. Open https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Open a new Kaggle notebook and attach the dataset in **Add Input**.
3. Import `notebooks/01_HAM10000_Seven_Class_Data_Audit.ipynb`.
4. Turn on `RUN_FULL_IMAGE_AUDIT = True` and `RUN_EXACT_HASH_AUDIT = True` in Section 1 before final run.
5. **Run All**, in order. Section 16 exports `ham10000_audit/ham10000_master_manifest.csv`; Section 18 exports `ham10000_7class/seven_class_master_manifest.csv`, train/val/test manifests and `class_mapping.json`.
6. Inspect warnings and do not silently bypass integrity assertions. Confirm no cross-lesion content duplicates or investigate before model use.
7. Choose **Save Version → Save & Run All**. After completion, inspect notebook-version **Output** to make sure the CSVs were saved.
8. Add the Kaggle version URL and date to the repo README or audit findings.

### If Kaggle session restarts
`/kaggle/working` is temporary and may be cleared. Re-run the notebook, or download output assets from a **completed saved version**. Never assume a file exists after a session reset.

### Before EfficientNet training
- Use the saved manifests; each image points to original JPEG files. If running in a different folder layout, rebuild paths from `image_relpath` instead of hard-coding an old session path.
- Keep `class_mapping.json` fixed in all model runs.
- Keep test partition locked until final evaluation.
- Notebook uses CPU for EDA; enable Kaggle GPU for EfficientNet training **later**.
