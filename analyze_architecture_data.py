from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def load_and_clean_data(filepath):
    filepath = _resolve_existing_path(filepath)
    df = pd.read_csv(filepath)
    
    column_mapping = {
        "חותמת זמן": "Timestamp",
        "מין": "Gender",
        "גיל": "Age",
        "מוסד לימודים": "Institution",
        "סטטוס תעסוקתי": "Employment_Status",
        "שנת לימודים נוכחית": "Student_Year",
        "שנות ניסיון במשרת סטודנט": "Student_Exp_Years",
        "שכר סטודנטיאלי נוכחי (שעתי)": "Student_Hourly_Wage",
        "השכלה": "Degree",
        "שנות ניסיון במקצוע (כולל תקופת משרת סטודנט)": "Professional_Exp",
        "במידה וסיימת את התואר בשנתיים האחרונות, מה היה השכר הסטודנטיאלי השעתי האחרון שלך?": "Last_Student_Wage",
        "השכר החודשי שלך ברוטו (כולל בונוסים, במידה וישנם):": "Monthly_Salary",
        "מספר שעות עבודה ביום (בממוצע)": "Daily_Hours",
        "מספר ימי עבודה בשבוע (בממוצע)": "Weekly_Days",
        "אופי העסקה": "Work_Mode",
        "מה התדירות שבה אתם מתבקשים לעבוד שעות נוספות?": "Overtime_Freq",
        "קרן השתלמות": "Keren_Hishtalmut",
        "סיבוס/תן-ביס/מימון ארוחות": "Meals",
        "רכב חברה": "Car",
        "במידה וישנם תנאים אחרים ניתן לפרט כאן (בונוסים שנתיים, ימי חופשה נוספים מעבר לחוק, ביטוח בריאות פרטי...)": "Other_Benefits",
        "מיקום גיאוררפי של מקום העבודה": "Location",
        "כמה עובדים במשרד?": "Company_Size",
        "באיזה תחום עיקרי המשרד עוסק/ת? (ניתן לבחור יותר מאחת)": "Field",
        "שביעות רצון מהמשרד": "Satisfaction",
        "שם המשרד": "Office_Name"
    }
    
    df = df.rename(columns=column_mapping)
    
    def parse_salary(val):
        if pd.isna(val) or val == "": return np.nan
        val = str(val).replace('ש"ח', '').replace(',', '').strip()
        range_match = re.match(r'(\d+)\s*-\s*(\d+)', val)
        if range_match: return (int(range_match.group(1)) + int(range_match.group(2))) / 2
        less_match = re.search(r'פחות מ-(\d+)', val)
        if less_match: return int(less_match.group(1)) - 500
        try: return float(val)
        except: return np.nan

    df['Monthly_Salary_Numeric'] = df['Monthly_Salary'].apply(parse_salary)
    df['Satisfaction'] = pd.to_numeric(df['Satisfaction'], errors='coerce')

    # Hours / days are strong signals and are fully filled in the dataset.
    # Keep them numeric for modeling.
    if 'Daily_Hours' in df.columns:
        df['Daily_Hours'] = pd.to_numeric(df['Daily_Hours'], errors='coerce')
    if 'Weekly_Days' in df.columns:
        df['Weekly_Days'] = pd.to_numeric(df['Weekly_Days'], errors='coerce')
    
    return df

def map_features(df):
    # Gender
    df['Gender_En'] = df['Gender'].map({"זכר": "Male", "נקבה": "Female"})
    
    # Institution
    def map_inst(val):
        if pd.isna(val): return "Other"
        if "טכניון" in val: return "Technion"
        if "תל אביב" in val: return "Tel Aviv Univ"
        if "בצלאל" in val: return "Bezalel"
        if "אריאל" in val: return "Ariel Univ"
        if "ויצו" in val: return "Wizo Haifa"
        return "Other"
    df['Institution_En'] = df['Institution'].apply(map_inst)
    
    # Experience (Updated mapping for better granularity if possible)
    exp_map = {
        "ללא ניסיון": 0, 
        "פחות משנתיים": 1, 
        "2-5 שנים": 3.5, 
        "5-10 שנים": 7.5, 
        "מעל 10 שנים": 12
    }
    df['Professional_Exp_Mapped'] = df['Professional_Exp'].apply(lambda x: exp_map.get(str(x).strip(), 0))
    
    # Company Size
    size_map = {"1-10 עובדים": 5, "11-30 עובדים": 20, "31-50 עובדים": 40, "50+ עובדים": 60}
    df['Company_Size_Mapped'] = df['Company_Size'].apply(lambda x: size_map.get(str(x).strip(), 10))

    # Location
    def map_loc(val):
        if pd.isna(val): return "Other"
        if "מרכז" in val: return "Center"
        if "תל אביב" in val: return "Center"
        if "חיפה" in val: return "Haifa"
        if "שרון" in val: return "Sharon"
        if "ירושלים" in val: return "Jerusalem"
        if "צפון" in val: return "North"
        if "דרום" in val: return "South"
        return "Other"
    df['Location_En'] = df['Location'].apply(map_loc)

    # Degree Level
    def map_degree(val):
        if pd.isna(val): return "Bachelor"
        if "שני" in str(val): return "Master"
        return "Bachelor"
    df['Degree_Level'] = df['Degree'].apply(map_degree)

    # Work mode (Hebrew -> English) for consistent UI/ML categories
    work_mode_map = {
        "פיזית בלבד": "On-site",
        "היברידית": "Hybrid",
        "מקוונת בלבד": "Remote",
    }
    if 'Work_Mode' in df.columns:
        df['Work_Mode_En'] = df['Work_Mode'].map(work_mode_map).fillna("Other")

    # Overtime frequency (Hebrew -> English)
    overtime_map = {
        "אף פעם": "Never",
        "לעיתים רחוקות מאוד": "Rarely",
        "מידי חודש": "Monthly",
        "מידי שבוע": "Weekly",
    }
    if 'Overtime_Freq' in df.columns:
        df['Overtime_Freq_En'] = df['Overtime_Freq'].map(overtime_map).fillna("Other")

    # Benefits (Hebrew -> English)
    yn_map = {"כן": "Yes", "לא": "No"}
    if 'Keren_Hishtalmut' in df.columns:
        df['Keren_Hishtalmut_En'] = df['Keren_Hishtalmut'].map(yn_map).fillna("No")
    if 'Meals' in df.columns:
        df['Meals_En'] = df['Meals'].map(yn_map).fillna("No")
    if 'Car' in df.columns:
        df['Car_En'] = df['Car'].map(yn_map).fillna("No")
    
    # Field extraction
    # We will create binary columns for key fields
    df['Field'] = df['Field'].astype(str)
    
    fields = {
        'Field_Residential': ["מגורים", "בנייה רוויה", "וילות"],
        'Field_Public': ["מבני ציבור", "חינוך"],
        'Field_TAMA38': ["תמ\"א", "התחדשות עירונית"],
        'Field_Interior': ["פנים"],
        'Field_Commercial': ["מסחר", "משרדים"],
        'Field_ProjectManagement': ["ניהול", "פיקוח"],
        'Field_Landscape': ["נוף"]
    }
    
    for field_col, keywords in fields.items():
        df[field_col] = df['Field'].apply(lambda x: 1 if any(k in x for k in keywords) else 0)

    return df

def train_model(
    df: pd.DataFrame,
    *,
    cv_splits: int = 5,
    random_state: int = 42,
    n_iter: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a stronger model using a proper sklearn Pipeline and 5-fold CV hyperparameter search.

    Returns a model *bundle* dict that includes:
    - model: fitted sklearn estimator (TransformedTargetRegressor wrapping a Pipeline)
    - feature_columns: raw feature columns expected at inference
    - defaults: default values used to fill missing inputs
    - metrics: CV metrics
    - uncertainty: absolute-error quantiles from out-of-fold predictions
    """
    # Filter valid data
    ml_df = df[df['Monthly_Salary_Numeric'].notna() & (df['Employment_Status'] == "משרה חלקית/מלאה")].copy()
    ml_df = ml_df[ml_df['Monthly_Salary_Numeric'] > 4000]

    # Feature groups
    cat_features = [
        'Gender_En',
        'Age',
        'Institution_En',
        'Location_En',
        'Degree_Level',
        'Work_Mode_En',
        'Overtime_Freq_En',
        'Keren_Hishtalmut_En',
        'Meals_En',
        # 'Car_En',  # extremely rare "Yes" in the dataset; tends to add noise
    ]

    num_features = [
        'Professional_Exp_Mapped',
        'Company_Size_Mapped',
        'Daily_Hours',
        'Weekly_Days',
    ]

    field_features = [c for c in ml_df.columns if c.startswith('Field_')]

    feature_columns = cat_features + num_features + field_features
    target = 'Monthly_Salary_Numeric'

    # Build X/y
    X = ml_df[feature_columns].copy()
    y = ml_df[target].astype(float)

    # Reasonable defaults for inference (used by GUI/CLI when optional fields are omitted)
    defaults: Dict[str, Any] = {}
    for c in cat_features:
        mode = ml_df[c].dropna().mode()
        defaults[c] = mode.iloc[0] if len(mode) else "Other"
    for c in num_features:
        defaults[c] = float(ml_df[c].median()) if ml_df[c].notna().any() else 0.0
    for c in field_features:
        defaults[c] = 0

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_features + field_features,
            ),
        ],
        remainder="drop",
    )

    base_reg = HistGradientBoostingRegressor(random_state=random_state)
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", base_reg)])

    # Log-transform target tends to help on salary-like distributions
    reg = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    param_distributions = {
        "regressor__model__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "regressor__model__max_depth": [None, 3, 4, 5, 6],
        "regressor__model__max_leaf_nodes": [15, 31, 63, 127],
        "regressor__model__min_samples_leaf": [10, 20, 30, 40],
        "regressor__model__l2_regularization": [0.0, 0.1, 1.0, 5.0, 10.0],
        "regressor__model__max_iter": [250, 400, 600, 900],
        "regressor__model__max_bins": [64, 128, 255],
    }

    search = RandomizedSearchCV(
        estimator=reg,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    search.fit(X, y)

    best_model = search.best_estimator_
    best_rmse = -float(search.best_score_)
    best_std = float(search.cv_results_["std_test_score"][search.best_index_])

    # Out-of-fold residuals for a pragmatic uncertainty band
    oof_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)
    abs_err = np.abs(y.to_numpy() - oof_pred)
    q50, q80, q90 = np.quantile(abs_err, [0.50, 0.80, 0.90])

    if verbose:
        print("\n=== Improved Model Performance (5-Fold CV + Hyperparameter Search) ===")
        print(f"Rows used: {len(X)}")
        print(f"Best CV RMSE: {best_rmse:,.2f} NIS")
        print(f"CV RMSE Std: {best_std:,.2f} NIS")
        print(f"Typical absolute error (median): {q50:,.0f} NIS")
        print(f"80% absolute error band: ±{q80:,.0f} NIS")

    return {
        "version": 1,
        "model": best_model,
        "feature_columns": feature_columns,
        "defaults": defaults,
        "metrics": {
            "cv_splits": cv_splits,
            "rows_used": int(len(X)),
            "cv_rmse": float(best_rmse),
            "cv_rmse_std": float(best_std),
        },
        "uncertainty": {
            "abs_err_p50": float(q50),
            "abs_err_p80": float(q80),
            "abs_err_p90": float(q90),
        },
        "best_params": search.best_params_,
    }

def predict_salary(model_bundle: Dict[str, Any], inputs: Dict[str, Any]) -> float:
    """
    Predict salary using the trained model bundle.

    `inputs` should contain raw (pre-onehot) feature values:
    - categorical strings (e.g. Gender_En="Female")
    - numeric values (e.g. Professional_Exp_Mapped=2.5)
    - Field_* binary flags (0/1)
    Missing inputs are filled from bundle defaults.
    """
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    defaults = model_bundle.get("defaults", {})

    row: Dict[str, Any] = {}
    for c in feature_columns:
        row[c] = inputs.get(c, defaults.get(c))

    X_one = pd.DataFrame([row], columns=feature_columns)
    pred = float(model.predict(X_one)[0])
    return pred


def predict_salary_with_range(model_bundle: Dict[str, Any], inputs: Dict[str, Any], *, band: str = "p80") -> Dict[str, float]:
    """
    Returns prediction + a simple uncertainty range using CV out-of-fold absolute-error quantiles.
    band: "p50" | "p80" | "p90"
    """
    pred = predict_salary(model_bundle, inputs)
    q = model_bundle.get("uncertainty", {}).get(f"abs_err_{band}", None)
    if q is None:
        return {"pred": pred}
    return {"pred": pred, "low": max(0.0, pred - float(q)), "high": pred + float(q)}


def _resolve_existing_path(path_like: str | Path) -> str:
    p = Path(path_like)
    if p.is_file():
        return str(p)
    alt = Path(__file__).with_name(str(path_like))
    if alt.is_file():
        return str(alt)
    return str(p)


def _file_md5(path_like: str | Path) -> str:
    p = Path(path_like)
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_or_train_model(
    *,
    data_path: str | Path = "data.csv",
    cache_path: str | Path = "salary_model.joblib",
    force_retrain: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load a cached model bundle if it matches the current data.csv content; otherwise train+cache.
    """
    data_path = _resolve_existing_path(data_path)
    cache_p = Path(cache_path)
    if not cache_p.is_absolute():
        cache_p = Path(__file__).with_name(str(cache_p))

    data_md5 = _file_md5(data_path)

    if (not force_retrain) and cache_p.is_file():
        try:
            bundle = joblib.load(cache_p)
            if bundle.get("data_md5") == data_md5 and bundle.get("version") == 1:
                if verbose:
                    m = bundle.get("metrics", {})
                    print("\nLoaded cached model.")
                    print(f"CV RMSE: {m.get('cv_rmse', float('nan')):,.2f} NIS (std {m.get('cv_rmse_std', float('nan')):,.2f})")
                return bundle
        except Exception:
            # Cache read failed; fall back to retrain
            pass

    if verbose:
        print("\nTraining model (first run or data changed)...")

    df = load_and_clean_data(data_path)
    df = map_features(df)
    bundle = train_model(df, verbose=verbose)
    bundle["data_md5"] = data_md5
    bundle["trained_at"] = datetime.now(timezone.utc).isoformat()

    try:
        joblib.dump(bundle, cache_p)
        if verbose:
            print(f"Saved model cache to: {cache_p}")
    except Exception as e:
        if verbose:
            print(f"Warning: failed to save cache ({e})")

    return bundle

def create_advanced_visualizations(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'

    graduates = df[df['Employment_Status'] == "משרה חלקית/מלאה"]
    
    # 1. Field Premium
    field_cols = [c for c in df.columns if c.startswith('Field_')]
    field_data = []
    for f in field_cols:
        # Get salaries where this field is active
        # Note: A person can have multiple fields, so this is overlapping
        sals = graduates[graduates[f] == 1]['Monthly_Salary_Numeric']
        if len(sals) > 5:
            field_data.append({'Field': f.replace('Field_', ''), 'Avg Salary': sals.mean()})
            
    if field_data:
        fdf = pd.DataFrame(field_data).sort_values('Avg Salary', ascending=False)
        plt.figure(figsize=(10, 6))
        # seaborn>=0.14: palette without hue is deprecated
        sns.barplot(x='Avg Salary', y='Field', hue='Field', data=fdf, palette='viridis', legend=False)
        plt.title('Average Salary by Architecture Field')
        plt.xlabel('Monthly Salary (NIS)')
        plt.tight_layout()
        out_path = Path(__file__).with_name('salary_by_field.png')
        plt.savefig(out_path)
        plt.close()
    
    # 2. Re-generate others
    # ... (Keep existing if needed, or focus on the new insight)

if __name__ == "__main__":
    bundle = get_or_train_model(data_path="data.csv", cache_path="salary_model.joblib", force_retrain=False, verbose=True)
    df = load_and_clean_data("data.csv")
    df = map_features(df)
    create_advanced_visualizations(df)
    print("\nVisualizations generated: salary_by_field.png")
