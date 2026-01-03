from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import altair as alt
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


def _graduates_df(df: pd.DataFrame) -> pd.DataFrame:
    grads = df[df["Employment_Status"] == "משרה חלקית/מלאה"].copy()
    if "Monthly_Salary_Numeric" in grads.columns:
        grads["Monthly_Salary_Numeric"] = pd.to_numeric(grads["Monthly_Salary_Numeric"], errors="coerce")
    if "Professional_Exp_Mapped" in grads.columns:
        grads["Professional_Exp_Mapped"] = pd.to_numeric(grads["Professional_Exp_Mapped"], errors="coerce")
    grads = grads[grads["Monthly_Salary_Numeric"].notna()].copy()
    return grads


def _cat_salary_table(grads: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in grads.columns:
        return pd.DataFrame(columns=[col, "Avg Salary", "Median Salary", "Count"])
    g = (
        grads.groupby(col, dropna=False)["Monthly_Salary_Numeric"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "Avg Salary", "median": "Median Salary", "count": "Count"})
        .sort_values("Avg Salary", ascending=False, ignore_index=True)
    )
    return g


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
    cache_exists = Path(cache_path).is_file()

    # Model load (cached)
    if "model_bundle" not in st.session_state:
        with st.spinner("Loading AI model (cached if available)…"):
            st.session_state["model_bundle"] = _load_model_cached(data_path, cache_path)

    bundle: Dict[str, Any] = st.session_state["model_bundle"]

    # Sidebar (status + actions)
    with st.sidebar:
        st.subheader("Model")

        if not cache_exists:
            st.info(
                "First run: no cached model found. The app will **train the model** automatically (may take ~1–2 minutes). "
                "Wait for the metrics below to appear before estimating salary.",
            )

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
        st.info(
            "Before estimating: make sure the **model is loaded/trained** (see the sidebar metrics). "
            "On the first run, training may take a minute.",
            icon="ℹ️",
        )

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
            st.caption(
                "Disclaimer: This is an **estimate** based on limited survey data and typical model error. "
                "Your real salary may differ significantly."
            )
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

        grads = _graduates_df(df)

        st.caption(
            f"Insights are computed from **{len(grads)}** employed respondents with a numeric salary in the dataset."
        )

        # Filters (lightweight, optional)
        with st.expander("Filters (optional)", expanded=False):
            f1, f2, f3 = st.columns(3)
            with f1:
                loc_filter = st.multiselect(
                    "Location",
                    sorted([x for x in grads.get("Location_En", pd.Series(dtype=str)).dropna().unique().tolist()]),
                    default=[],
                )
            with f2:
                mode_filter = st.multiselect(
                    "Work mode",
                    sorted([x for x in grads.get("Work_Mode_En", pd.Series(dtype=str)).dropna().unique().tolist()]),
                    default=[],
                )
            with f3:
                degree_filter = st.multiselect(
                    "Degree",
                    sorted([x for x in grads.get("Degree_Level", pd.Series(dtype=str)).dropna().unique().tolist()]),
                    default=[],
                )

            filtered = grads.copy()
            if loc_filter and "Location_En" in filtered.columns:
                filtered = filtered[filtered["Location_En"].isin(loc_filter)]
            if mode_filter and "Work_Mode_En" in filtered.columns:
                filtered = filtered[filtered["Work_Mode_En"].isin(mode_filter)]
            if degree_filter and "Degree_Level" in filtered.columns:
                filtered = filtered[filtered["Degree_Level"].isin(degree_filter)]

            st.caption(f"Using **{len(filtered)}** rows after filters.")
        if "filtered" not in locals():
            filtered = grads

        # 1) Salary distribution
        st.markdown("**Salary distribution (gross monthly)**")
        hist = (
            alt.Chart(filtered)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Monthly_Salary_Numeric:Q", bin=alt.Bin(maxbins=30), title="Monthly salary (NIS)"),
                y=alt.Y("count():Q", title="Count"),
                tooltip=[alt.Tooltip("count():Q", title="Count")],
            )
            .properties(height=260)
        )
        st.altair_chart(hist, use_container_width=True)

        # 2) Salary vs experience
        st.markdown("**Salary vs years of experience**")
        exp_df = filtered[filtered["Professional_Exp_Mapped"].notna()].copy()
        scatter = (
            alt.Chart(exp_df)
            .mark_circle(size=48, opacity=0.35)
            .encode(
                x=alt.X("Professional_Exp_Mapped:Q", title="Years of experience"),
                y=alt.Y("Monthly_Salary_Numeric:Q", title="Monthly salary (NIS)"),
                tooltip=[
                    alt.Tooltip("Professional_Exp_Mapped:Q", title="Experience"),
                    alt.Tooltip("Monthly_Salary_Numeric:Q", title="Salary", format=",.0f"),
                    alt.Tooltip("Location_En:N", title="Location"),
                    alt.Tooltip("Work_Mode_En:N", title="Work mode"),
                ],
            )
            .properties(height=320)
        )
        trend = (
            alt.Chart(exp_df)
            .transform_regression("Professional_Exp_Mapped", "Monthly_Salary_Numeric")
            .mark_line(color="#3b82f6", strokeWidth=3)
            .encode(x="Professional_Exp_Mapped:Q", y="Monthly_Salary_Numeric:Q")
        )
        st.altair_chart(scatter + trend, use_container_width=True)

        # 3) Category comparisons
        st.markdown("**Category comparisons**")
        c_left, c_right = st.columns(2, gap="large")

        with c_left:
            st.markdown("**By location (box plot)**")
            if "Location_En" in filtered.columns and filtered["Location_En"].notna().any():
                loc_box = (
                    alt.Chart(filtered.dropna(subset=["Location_En"]))
                    .mark_boxplot(size=24)
                    .encode(
                        x=alt.X(
                            "Location_En:N",
                            sort=alt.SortField("Monthly_Salary_Numeric", op="median", order="descending"),
                            title="Location",
                        ),
                        y=alt.Y("Monthly_Salary_Numeric:Q", title="Monthly salary (NIS)"),
                        tooltip=[alt.Tooltip("Location_En:N", title="Location")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(loc_box, use_container_width=True)
            else:
                st.info("Location data not available for chart.")

        with c_right:
            st.markdown("**By work mode (avg + count)**")
            mode_tbl = _cat_salary_table(filtered, "Work_Mode_En")
            if not mode_tbl.empty:
                mode_bar = (
                    alt.Chart(mode_tbl)
                    .mark_bar()
                    .encode(
                        x=alt.X("Work_Mode_En:N", sort="-y", title="Work mode"),
                        y=alt.Y("Avg Salary:Q", title="Average salary (NIS)"),
                        tooltip=[
                            alt.Tooltip("Work_Mode_En:N", title="Work mode"),
                            alt.Tooltip("Avg Salary:Q", title="Avg", format=",.0f"),
                            alt.Tooltip("Median Salary:Q", title="Median", format=",.0f"),
                            alt.Tooltip("Count:Q", title="Count"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(mode_bar, use_container_width=True)
            else:
                st.info("Work mode data not available for chart.")

        # 4) Benefits comparisons
        st.markdown("**Benefits impact (avg salary)**")
        b1, b2 = st.columns(2, gap="large")
        with b1:
            keren_tbl = _cat_salary_table(filtered, "Keren_Hishtalmut_En")
            if not keren_tbl.empty:
                keren_bar = (
                    alt.Chart(keren_tbl)
                    .mark_bar()
                    .encode(
                        x=alt.X("Keren_Hishtalmut_En:N", title="Keren Hishtalmut"),
                        y=alt.Y("Avg Salary:Q", title="Average salary (NIS)"),
                        tooltip=[
                            alt.Tooltip("Keren_Hishtalmut_En:N", title="Keren"),
                            alt.Tooltip("Avg Salary:Q", title="Avg", format=",.0f"),
                            alt.Tooltip("Count:Q", title="Count"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(keren_bar, use_container_width=True)
            else:
                st.info("Keren Hishtalmut data not available for chart.")

        with b2:
            meals_tbl = _cat_salary_table(filtered, "Meals_En")
            if not meals_tbl.empty:
                meals_bar = (
                    alt.Chart(meals_tbl)
                    .mark_bar()
                    .encode(
                        x=alt.X("Meals_En:N", title="Meals"),
                        y=alt.Y("Avg Salary:Q", title="Average salary (NIS)"),
                        tooltip=[
                            alt.Tooltip("Meals_En:N", title="Meals"),
                            alt.Tooltip("Avg Salary:Q", title="Avg", format=",.0f"),
                            alt.Tooltip("Count:Q", title="Count"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(meals_bar, use_container_width=True)
            else:
                st.info("Meals data not available for chart.")

        # 5) Field premium (existing)
        st.markdown("**Average salary by field (overlapping groups)**")
        field_tbl = _field_salary_table(df)
        if field_tbl.empty:
            st.info("No field chart available (not enough samples per field).")
        else:
            st.bar_chart(field_tbl.set_index("Field")["Avg Salary"], height=320)
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
        st.warning(
            "Disclaimer: The dataset is relatively small and may not represent the full market. "
            "Outputs are **estimates** and can be **not accurate**.",
            icon="⚠️",
        )

        st.divider()
        st.markdown("**Credit:** Omer Taub")
        st.markdown("**Contact:** [linkedin.com/in/omertaub](https://www.linkedin.com/in/omertaub/)")


if __name__ == "__main__":
    main()


