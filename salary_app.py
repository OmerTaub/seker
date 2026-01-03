from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

import analyze_architecture_data as analysis


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data.csv"
MODEL_CACHE_PATH = APP_DIR / "salary_model.joblib"


COMPANY_SIZE_TO_NUM = {
    "1-10 Employees": 5,
    "11-30 Employees": 20,
    "31-50 Employees": 40,
    "50+ Employees": 60,
}

FIELD_LABEL_TO_KEY = {
    "Residential": "Field_Residential",
    "Public buildings": "Field_Public",
    'TAMA 38 / urban renewal': "Field_TAMA38",
    "Interior design": "Field_Interior",
    "Commercial / offices": "Field_Commercial",
    "Project management": "Field_ProjectManagement",
    "Landscape": "Field_Landscape",
}


def _nis(v: float) -> str:
    return f"₪ {v:,.0f}"


@st.cache_resource(show_spinner=False)
def _load_model_cached(data_path: str, cache_path: str) -> Dict[str, Any]:
    return analysis.get_or_train_model(
        data_path=data_path,
        cache_path=cache_path,
        force_retrain=False,
        verbose=False,
    )


@st.cache_data(show_spinner=False)
def _load_df_cached(data_path: str) -> pd.DataFrame:
    df = analysis.load_and_clean_data(data_path)
    df = analysis.map_features(df)
    return df


def _field_salary_table(df: pd.DataFrame) -> pd.DataFrame:
    # Mirrors analyze_architecture_data.create_advanced_visualizations(), but produces a dataframe for Streamlit.
    grads = df[df["Employment_Status"] == "משרה חלקית/מלאה"].copy()
    if "Monthly_Salary_Numeric" not in grads.columns:
        return pd.DataFrame(columns=["Field", "Avg Salary", "Count"])

    rows = []
    for col in [c for c in grads.columns if c.startswith("Field_")]:
        sals = grads.loc[grads[col] == 1, "Monthly_Salary_Numeric"].dropna()
        if len(sals) >= 5:
            rows.append(
                {
                    "Field": col.replace("Field_", ""),
                    "Avg Salary": float(sals.mean()),
                    "Count": int(len(sals)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Field", "Avg Salary", "Count"])

    out = pd.DataFrame(rows).sort_values("Avg Salary", ascending=False, ignore_index=True)
    return out


def _ensure_paths() -> Tuple[str, str]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing data file: {DATA_PATH}")
    return str(DATA_PATH), str(MODEL_CACHE_PATH)


def main() -> None:
    st.set_page_config(
        page_title="Architect AI — Salary Estimator",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Architect AI — Salary Estimator")
    st.caption("A comfortable UI for estimating monthly gross salary (NIS) from the survey-trained model.")

    data_path, cache_path = _ensure_paths()

    # Model load (cached)
    if "model_bundle" not in st.session_state:
        with st.spinner("Loading AI model (cached if available)…"):
            st.session_state["model_bundle"] = _load_model_cached(data_path, cache_path)

    bundle: Dict[str, Any] = st.session_state["model_bundle"]

    # Sidebar (status + actions)
    with st.sidebar:
        st.subheader("Model")

        m = bundle.get("metrics", {})
        u = bundle.get("uncertainty", {})
        trained_at = bundle.get("trained_at", "—")

        st.write(f"**Trained at:** {trained_at}")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{m.get('rows_used', '—')}")
        c2.metric("CV RMSE", _nis(float(m.get("cv_rmse", 0.0))) if m.get("cv_rmse") is not None else "—")

        st.caption(
            f"Typical abs error: median {_nis(float(u.get('abs_err_p50', 0.0)))} • "
            f"80% band ±{_nis(float(u.get('abs_err_p80', 0.0)))}"
        )

        col_a, col_b = st.columns(2)
        retrain = col_a.button("Retrain", use_container_width=True)
        reset = col_b.button("Reset UI", use_container_width=True)

        if reset:
            for k in ["model_bundle", "last_result"]:
                st.session_state.pop(k, None)
            st.rerun()

        if retrain:
            with st.spinner("Retraining model… this may take a minute."):
                st.session_state["model_bundle"] = analysis.get_or_train_model(
                    data_path=data_path,
                    cache_path=cache_path,
                    force_retrain=True,
                    verbose=False,
                )
            st.success("Model retrained.")
            st.rerun()

        st.divider()
        st.subheader("Quick help")
        st.write(
            "- Fill **hours/days** and pick relevant **fields** — these features improve accuracy.\n"
            "- Results are **estimates** based on the dataset and typical model error."
        )

    tabs = st.tabs(["Estimate", "Insights", "About"])

    # ---------------------------
    # Estimate tab
    # ---------------------------
    with tabs[0]:
        st.subheader("Estimate your monthly salary (gross)")

        with st.form("estimate_form", border=True):
            left, right = st.columns(2, gap="large")

            with left:
                st.markdown("**Personal**")
                gender = st.selectbox("Gender", ["Female", "Male"], index=0)
                age = st.selectbox("Age group", ["20-25", "26-30", "31-35", "36-40", "41-45"], index=1)
                degree = st.selectbox("Degree", ["Bachelor", "Master"], index=0)
                institution = st.selectbox(
                    "Institution",
                    ["Technion", "Tel Aviv Univ", "Bezalel", "Ariel Univ", "Wizo Haifa", "Other"],
                    index=5,
                )

            with right:
                st.markdown("**Job details**")
                years_exp = st.number_input("Years of experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)
                company_size_label = st.selectbox("Company size", list(COMPANY_SIZE_TO_NUM.keys()), index=1)
                location = st.selectbox(
                    "Location",
                    ["Center", "Haifa", "Jerusalem", "North", "South", "Sharon", "Other"],
                    index=0,
                )
                work_mode = st.selectbox("Work mode", ["On-site", "Hybrid", "Remote", "Other"], index=0)
                overtime = st.selectbox("Overtime frequency", ["Never", "Rarely", "Monthly", "Weekly", "Other"], index=0)

                daily_hours = st.slider("Daily hours", min_value=5, max_value=10, value=9, step=1)
                weekly_days = st.slider("Days / week", min_value=2, max_value=5, value=5, step=1)

            st.markdown("**Field(s)**")
            selected_fields = st.multiselect(
                "Select all relevant fields (multi-select)",
                list(FIELD_LABEL_TO_KEY.keys()),
                default=[],
            )

            st.markdown("**Benefits & extras**")
            b1, b2 = st.columns(2)
            with b1:
                keren = st.checkbox("Keren Hishtalmut", value=False)
            with b2:
                meals = st.checkbox("Meals (Cibus/TenBis)", value=False)

            submitted = st.form_submit_button("Calculate", use_container_width=True)

        if submitted:
            profile: Dict[str, Any] = {
                "Gender_En": gender,
                "Age": age,
                "Institution_En": institution,
                "Degree_Level": degree,
                "Professional_Exp_Mapped": float(years_exp),
                "Company_Size_Mapped": COMPANY_SIZE_TO_NUM.get(company_size_label, 20),
                "Location_En": location,
                "Work_Mode_En": work_mode,
                "Overtime_Freq_En": overtime,
                "Daily_Hours": float(daily_hours),
                "Weekly_Days": float(weekly_days),
                "Keren_Hishtalmut_En": "Yes" if keren else "No",
                "Meals_En": "Yes" if meals else "No",
            }

            for label, key in FIELD_LABEL_TO_KEY.items():
                profile[key] = 1 if label in set(selected_fields) else 0

            out = analysis.predict_salary_with_range(bundle, profile, band="p80")
            st.session_state["last_result"] = out

        if "last_result" in st.session_state:
            out = st.session_state["last_result"]
            pred = float(out.get("pred", 0.0))

            st.divider()
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.metric("Estimated monthly salary", _nis(pred))

            low = out.get("low")
            high = out.get("high")
            if low is not None and high is not None:
                c2.metric("Low (80%)", _nis(float(low)))
                c3.metric("High (80%)", _nis(float(high)))
            else:
                c2.metric("Low (80%)", "—")
                c3.metric("High (80%)", "—")

    # ---------------------------
    # Insights tab
    # ---------------------------
    with tabs[1]:
        st.subheader("Model insights")

        c1, c2, c3, c4 = st.columns(4)
        m = bundle.get("metrics", {})
        u = bundle.get("uncertainty", {})

        c1.metric("Rows used", f"{m.get('rows_used', '—')}")
        c2.metric("CV RMSE", _nis(float(m.get("cv_rmse", 0.0))) if m.get("cv_rmse") is not None else "—")
        c3.metric("Median abs error", _nis(float(u.get("abs_err_p50", 0.0))))
        c4.metric("80% abs error band", f"±{_nis(float(u.get('abs_err_p80', 0.0)))}")

        with st.spinner("Loading dataset…"):
            df = _load_df_cached(data_path)

        field_tbl = _field_salary_table(df)
        if field_tbl.empty:
            st.info("No field chart available (not enough samples per field).")
        else:
            st.markdown("**Average salary by field (overlapping groups)**")
            st.bar_chart(field_tbl.set_index("Field")["Avg Salary"], height=380)
            st.dataframe(
                field_tbl.style.format({"Avg Salary": "₪ {:,.0f}"}),
                use_container_width=True,
                hide_index=True,
            )

    # ---------------------------
    # About tab
    # ---------------------------
    with tabs[2]:
        st.subheader("About")
        st.write(
            "This app uses a cross-validated, hyperparameter-tuned model trained on architecture survey data.\n\n"
            "The estimate is **monthly gross salary (NIS)**. The range is derived from out-of-fold CV absolute-error "
            "quantiles (so it reflects the model’s typical error on similar data)."
        )


if __name__ == "__main__":
    main()


