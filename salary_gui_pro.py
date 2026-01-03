import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image

import analyze_architecture_data as analysis

# Global look & feel
ctk.set_appearance_mode("Dark")  # "System" | "Dark" | "Light"
ctk.set_default_color_theme("dark-blue")


class SalaryCalculatorPro(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Architect AI — Salary Estimator Pro")
        self.geometry("1100x760")
        self.minsize(1020, 700)

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Model state
        self.bundle = None
        self.df = None
        self.is_loading = True

        # UI
        self._build_sidebar()
        self._build_main()

        # Start background load
        self._set_loading(True, "Loading AI model (cached if available)…")
        threading.Thread(target=self._load_model_background, daemon=True).start()

    # -----------------------------
    # UI construction
    # -----------------------------
    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="Architect AI",
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(22, 4), sticky="w")

        self.tagline_label = ctk.CTkLabel(
            self.sidebar,
            text="CV-tuned salary predictor",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray60"),
        )
        self.tagline_label.grid(row=1, column=0, padx=20, pady=(0, 18), sticky="w")

        self.progress = ctk.CTkProgressBar(self.sidebar, mode="indeterminate")
        self.progress.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Starting…",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray60"),
            justify="left",
            anchor="w",
        )
        self.status_label.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.metrics_label = ctk.CTkLabel(
            self.sidebar,
            text="Model: —",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray60"),
            justify="left",
            anchor="w",
        )
        self.metrics_label.grid(row=4, column=0, padx=20, pady=(0, 14), sticky="ew")

        self.btn_retrain = ctk.CTkButton(
            self.sidebar,
            text="Retrain model",
            height=36,
            fg_color=("#0f172a", "#111827"),
            hover_color=("#111827", "#0b1220"),
            command=self._retrain_clicked,
        )
        self.btn_retrain.grid(row=5, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Theme", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="ew")
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
        )
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 20), sticky="s ew")
        self.appearance_mode_optionemenu.set("Dark")

    def _build_main(self):
        self.tabs = ctk.CTkTabview(self, fg_color="transparent")
        self.tabs.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.tabs.add("Estimate")
        self.tabs.add("Insights")
        self.tabs.add("About")

        # Estimate tab (scrollable)
        self.estimate = ctk.CTkScrollableFrame(self.tabs.tab("Estimate"), fg_color="transparent")
        self.estimate.pack(fill="both", expand=True)
        self.estimate.grid_columnconfigure(0, weight=1)
        self.estimate.grid_columnconfigure(1, weight=1)

        header = ctk.CTkLabel(
            self.estimate,
            text="Estimate your monthly salary (gross)",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header.grid(row=0, column=0, columnspan=2, padx=8, pady=(6, 16), sticky="w")

        # Cards
        self.card_personal = self._card(self.estimate, "Personal", row=1, col=0)
        self.card_job = self._card(self.estimate, "Job details", row=1, col=1)
        self.card_fields = self._card(self.estimate, "Field(s)", row=2, col=0, col_span=2)
        self.card_benefits = self._card(self.estimate, "Benefits & extras", row=3, col=0, col_span=2)

        # Personal inputs
        self.gender_var = self._dropdown(self.card_personal, "Gender", ["Female", "Male"], default="Female")
        self.age_var = self._dropdown(self.card_personal, "Age group", ["20-25", "26-30", "31-35", "36-40", "41-45"], default="26-30")
        self.degree_var = self._dropdown(self.card_personal, "Degree", ["Bachelor", "Master"], default="Bachelor")
        self.inst_var = self._dropdown(
            self.card_personal,
            "Institution",
            ["Technion", "Tel Aviv Univ", "Bezalel", "Ariel Univ", "Wizo Haifa", "Other"],
            default="Other",
        )

        # Job inputs
        self.exp_entry = ctk.CTkEntry(self.card_job, placeholder_text="e.g. 2.5")
        self._labeled_widget(self.card_job, "Years of experience", self.exp_entry)
        self.exp_entry.insert(0, "1")

        self.size_var = self._dropdown(
            self.card_job,
            "Company size",
            ["1-10 Employees", "11-30 Employees", "31-50 Employees", "50+ Employees"],
            default="11-30 Employees",
        )
        self.loc_var = self._dropdown(
            self.card_job,
            "Location",
            ["Center", "Haifa", "Jerusalem", "North", "South", "Sharon", "Other"],
            default="Center",
        )
        self.work_mode_var = self._dropdown(self.card_job, "Work mode", ["On-site", "Hybrid", "Remote"], default="On-site")
        self.overtime_var = self._dropdown(self.card_job, "Overtime frequency", ["Never", "Rarely", "Monthly", "Weekly"], default="Never")

        # Hours / days sliders
        self.daily_hours_val = tk.DoubleVar(value=9.0)
        self.weekly_days_val = tk.DoubleVar(value=5.0)
        self._slider_row(self.card_job, "Daily hours", self.daily_hours_val, from_=5, to=10, step=1)
        self._slider_row(self.card_job, "Days / week", self.weekly_days_val, from_=2, to=5, step=1)

        # Fields (multi-select)
        self.field_vars = {
            "Field_Residential": tk.IntVar(value=0),
            "Field_Public": tk.IntVar(value=0),
            "Field_TAMA38": tk.IntVar(value=0),
            "Field_Interior": tk.IntVar(value=0),
            "Field_Commercial": tk.IntVar(value=0),
            "Field_ProjectManagement": tk.IntVar(value=0),
            "Field_Landscape": tk.IntVar(value=0),
        }
        self._build_fields_grid(self.card_fields)

        # Benefits toggles
        self.keren_var = tk.IntVar(value=0)
        self.meals_var = tk.IntVar(value=0)
        self._switch(self.card_benefits, "Keren Hishtalmut", self.keren_var)
        self._switch(self.card_benefits, "Meals (Cibus/TenBis)", self.meals_var)

        # Action buttons
        self.btn_calculate = ctk.CTkButton(
            self.estimate,
            text="CALCULATE",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=52,
            fg_color="#3b82f6",
            hover_color="#2563eb",
            state="disabled",
            command=self.calculate_event,
        )
        self.btn_calculate.grid(row=4, column=0, columnspan=2, padx=10, pady=(18, 10), sticky="ew")

        # Result card
        self.card_result = ctk.CTkFrame(self.estimate, fg_color=("gray95", "gray12"))
        self.card_result.grid(row=5, column=0, columnspan=2, padx=10, pady=(10, 18), sticky="ew")
        self.card_result.grid_columnconfigure(0, weight=1)

        self.lbl_result_title = ctk.CTkLabel(
            self.card_result,
            text="Estimated monthly salary",
            font=ctk.CTkFont(size=13),
            text_color=("gray35", "gray70"),
        )
        self.lbl_result_title.grid(row=0, column=0, padx=18, pady=(18, 2), sticky="w")

        self.lbl_result_val = ctk.CTkLabel(
            self.card_result,
            text="₪ —",
            font=ctk.CTkFont(size=44, weight="bold"),
            text_color="#10b981",
        )
        self.lbl_result_val.grid(row=1, column=0, padx=18, pady=(0, 2), sticky="w")

        self.lbl_result_range = ctk.CTkLabel(
            self.card_result,
            text="Typical range: —",
            font=ctk.CTkFont(size=12),
            text_color=("gray35", "gray70"),
        )
        self.lbl_result_range.grid(row=2, column=0, padx=18, pady=(0, 16), sticky="w")

        # Insights tab
        self.insights = ctk.CTkFrame(self.tabs.tab("Insights"), fg_color="transparent")
        self.insights.pack(fill="both", expand=True)
        self.insights.grid_columnconfigure(0, weight=1)

        self.insights_title = ctk.CTkLabel(self.insights, text="Model insights", font=ctk.CTkFont(size=22, weight="bold"))
        self.insights_title.grid(row=0, column=0, padx=8, pady=(6, 12), sticky="w")

        self.insights_stats = ctk.CTkLabel(
            self.insights,
            text="Loading model stats…",
            font=ctk.CTkFont(size=13),
            justify="left",
            anchor="w",
            text_color=("gray35", "gray70"),
        )
        self.insights_stats.grid(row=1, column=0, padx=8, pady=(0, 12), sticky="ew")

        self.chart_frame = ctk.CTkFrame(self.insights, fg_color=("gray95", "gray12"))
        self.chart_frame.grid(row=2, column=0, padx=8, pady=(0, 12), sticky="nsew")
        self.chart_frame.grid_columnconfigure(0, weight=1)

        self.chart_label = ctk.CTkLabel(self.chart_frame, text="Generating chart…", text_color=("gray40", "gray60"))
        self.chart_label.grid(row=0, column=0, padx=18, pady=18, sticky="w")

        self.btn_refresh_insights = ctk.CTkButton(
            self.insights,
            text="Refresh chart",
            height=36,
            command=self._refresh_insights_clicked,
        )
        self.btn_refresh_insights.grid(row=3, column=0, padx=8, pady=(0, 8), sticky="w")

        # About tab
        self.about = ctk.CTkFrame(self.tabs.tab("About"), fg_color="transparent")
        self.about.pack(fill="both", expand=True)
        about_text = (
            "Architect AI uses a cross-validated, hyperparameter-tuned model trained on survey data.\n\n"
            "Tip: Fill hours/days and select relevant fields — those features noticeably improve accuracy."
        )
        self.about_label = ctk.CTkLabel(self.about, text=about_text, justify="left", anchor="nw")
        self.about_label.pack(fill="both", expand=True, padx=12, pady=12)

    # -----------------------------
    # UI helpers
    # -----------------------------
    def _card(self, parent, title: str, *, row: int, col: int, col_span: int = 1):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=col, columnspan=col_span, padx=10, pady=10, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        lbl = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=16, weight="bold"))
        lbl.grid(row=0, column=0, padx=18, pady=(16, 8), sticky="w")
        frame._next_row = 1  # type: ignore[attr-defined]
        return frame

    def _dropdown(self, parent, label: str, values, *, default=None):
        r = parent._next_row  # type: ignore[attr-defined]
        parent._next_row += 2  # type: ignore[attr-defined]

        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12), text_color=("gray35", "gray70")).grid(
            row=r, column=0, padx=18, pady=(6, 0), sticky="w"
        )
        var = tk.StringVar(value=default if default is not None else values[0])
        cb = ctk.CTkComboBox(parent, values=values, variable=var, state="readonly")
        cb.grid(row=r + 1, column=0, padx=18, pady=(6, 10), sticky="ew")
        return var

    def _labeled_widget(self, parent, label: str, widget):
        r = parent._next_row  # type: ignore[attr-defined]
        parent._next_row += 2  # type: ignore[attr-defined]

        ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12), text_color=("gray35", "gray70")).grid(
            row=r, column=0, padx=18, pady=(6, 0), sticky="w"
        )
        widget.grid(row=r + 1, column=0, padx=18, pady=(6, 10), sticky="ew")

    def _slider_row(self, parent, label: str, var: tk.DoubleVar, *, from_: int, to: int, step: int = 1):
        r = parent._next_row  # type: ignore[attr-defined]
        parent._next_row += 2  # type: ignore[attr-defined]

        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.grid(row=r, column=0, padx=18, pady=(6, 0), sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        value_label = ctk.CTkLabel(top, text=f"{label}: {int(var.get())}", text_color=("gray35", "gray70"))
        value_label.grid(row=0, column=0, sticky="w")

        def _on_change(v):
            var.set(round(float(v) / step) * step)
            value_label.configure(text=f"{label}: {int(var.get())}")

        slider = ctk.CTkSlider(parent, from_=from_, to=to, number_of_steps=(to - from_) // step, command=_on_change)
        slider.set(var.get())
        slider.grid(row=r + 1, column=0, padx=18, pady=(6, 10), sticky="ew")

    def _switch(self, parent, label: str, var: tk.IntVar):
        r = parent._next_row  # type: ignore[attr-defined]
        parent._next_row += 1  # type: ignore[attr-defined]

        sw = ctk.CTkSwitch(parent, text=label, variable=var, onvalue=1, offvalue=0)
        sw.grid(row=r, column=0, padx=18, pady=(6, 10), sticky="w")

    def _build_fields_grid(self, parent):
        grid = ctk.CTkFrame(parent, fg_color="transparent")
        r = parent._next_row  # type: ignore[attr-defined]
        parent._next_row += 1  # type: ignore[attr-defined]
        grid.grid(row=r, column=0, padx=18, pady=(6, 14), sticky="ew")
        for i in range(3):
            grid.grid_columnconfigure(i, weight=1)

        items = [
            ("Residential", "Field_Residential"),
            ("Public buildings", "Field_Public"),
            ("TAMA 38 / renewal", "Field_TAMA38"),
            ("Interior design", "Field_Interior"),
            ("Commercial / offices", "Field_Commercial"),
            ("Project management", "Field_ProjectManagement"),
            ("Landscape", "Field_Landscape"),
        ]
        for idx, (label, key) in enumerate(items):
            row = idx // 3
            col = idx % 3
            cb = ctk.CTkCheckBox(grid, text=label, variable=self.field_vars[key], onvalue=1, offvalue=0)
            cb.grid(row=row, column=col, padx=6, pady=6, sticky="w")

        hint = ctk.CTkLabel(
            parent,
            text="Tip: select all relevant fields — the dataset allows multiple fields per person.",
            font=ctk.CTkFont(size=12),
            text_color=("gray35", "gray70"),
        )
        hint.grid(row=parent._next_row, column=0, padx=18, pady=(0, 16), sticky="w")  # type: ignore[attr-defined]
        parent._next_row += 1  # type: ignore[attr-defined]

    # -----------------------------
    # Model lifecycle
    # -----------------------------
    def _set_loading(self, loading: bool, status_text: str):
        self.is_loading = loading
        self.status_label.configure(text=status_text, text_color=("gray40", "gray60"))
        if loading:
            self.progress.start()
            self.btn_calculate.configure(state="disabled")
            self.btn_retrain.configure(state="disabled")
        else:
            self.progress.stop()
            self.btn_calculate.configure(state="normal")
            self.btn_retrain.configure(state="normal")

    def _load_model_background(self, *, force_retrain: bool = False):
        try:
            bundle = analysis.get_or_train_model(
                data_path="data.csv",
                cache_path="salary_model.joblib",
                force_retrain=force_retrain,
                verbose=False,
            )
            df = analysis.load_and_clean_data("data.csv")
            df = analysis.map_features(df)

            # Generate chart (optional but nice)
            analysis.create_advanced_visualizations(df)

            self.after(0, lambda: self._on_model_loaded(bundle, df))
        except Exception as e:
            self.after(0, lambda: self._on_model_error(e))

    def _on_model_loaded(self, bundle, df):
        self.bundle = bundle
        self.df = df

        m = bundle.get("metrics", {})
        u = bundle.get("uncertainty", {})
        rmse = m.get("cv_rmse", None)
        rmse_std = m.get("cv_rmse_std", None)
        rows = m.get("rows_used", None)

        self.metrics_label.configure(
            text=f"CV RMSE: ₪ {rmse:,.0f} (±{rmse_std:,.0f})\nRows: {rows}\n80% error band: ±₪ {u.get('abs_err_p80', 0):,.0f}",
        )

        self.insights_stats.configure(
            text=(
                f"5-fold CV (hyperparameter search)\n"
                f"- Rows used: {rows}\n"
                f"- CV RMSE: ₪ {rmse:,.0f} (std ₪ {rmse_std:,.0f})\n"
                f"- Typical abs error (median): ₪ {u.get('abs_err_p50', 0):,.0f}\n"
                f"- 80% abs error band: ±₪ {u.get('abs_err_p80', 0):,.0f}"
            )
        )

        self._set_loading(False, "Model loaded — ready.")
        self.status_label.configure(text_color="#10b981")

        self._load_chart_image()

    def _on_model_error(self, e: Exception):
        self._set_loading(True, "Failed to load model.")
        self.progress.stop()
        self.status_label.configure(text=f"Error: {e}", text_color="red")
        messagebox.showerror("Model error", str(e))

    def _retrain_clicked(self):
        if self.is_loading:
            return
        if not messagebox.askyesno("Retrain model", "Retrain model now? This may take a minute."):
            return
        self._set_loading(True, "Retraining model…")
        threading.Thread(target=self._load_model_background, kwargs={"force_retrain": True}, daemon=True).start()

    def _refresh_insights_clicked(self):
        if self.is_loading or self.df is None:
            return
        try:
            analysis.create_advanced_visualizations(self.df)
            self._load_chart_image()
        except Exception as e:
            messagebox.showerror("Insights error", str(e))

    def _load_chart_image(self):
        chart_path = Path(__file__).with_name("salary_by_field.png")
        if not chart_path.is_file():
            self.chart_label.configure(text="No chart found yet.")
            return

        try:
            img = Image.open(chart_path)
            # Fit nicely within the frame
            target_w = 820
            ratio = img.height / img.width
            target_h = int(target_w * ratio)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(target_w, target_h))
            self._chart_img_ref = ctk_img  # keep reference
            self.chart_label.configure(image=ctk_img, text="")
        except Exception:
            self.chart_label.configure(text="Failed to load chart image.")

    # -----------------------------
    # Prediction
    # -----------------------------
    def calculate_event(self):
        if self.is_loading or self.bundle is None:
            return

        try:
            exp_val = float(self.exp_entry.get())
        except ValueError:
            self.lbl_result_val.configure(text="Invalid experience", text_color="red")
            return

        size_map = {"1-10 Employees": 5, "11-30 Employees": 20, "31-50 Employees": 40, "50+ Employees": 60}

        profile = {
            "Gender_En": self.gender_var.get(),
            "Age": self.age_var.get(),
            "Institution_En": self.inst_var.get(),
            "Degree_Level": self.degree_var.get(),
            "Professional_Exp_Mapped": exp_val,
            "Company_Size_Mapped": size_map.get(self.size_var.get(), 20),
            "Location_En": self.loc_var.get(),
            "Work_Mode_En": self.work_mode_var.get(),
            "Overtime_Freq_En": self.overtime_var.get(),
            "Daily_Hours": float(self.daily_hours_val.get()),
            "Weekly_Days": float(self.weekly_days_val.get()),
            "Keren_Hishtalmut_En": "Yes" if self.keren_var.get() else "No",
            "Meals_En": "Yes" if self.meals_var.get() else "No",
            **{k: int(v.get()) for k, v in self.field_vars.items()},
        }

        out = analysis.predict_salary_with_range(self.bundle, profile, band="p80")
        pred = out["pred"]

        self.lbl_result_val.configure(text_color="#10b981")
        self.animate_value(0, pred)

        if "low" in out and "high" in out:
            self.lbl_result_range.configure(text=f"Typical range (80%): ₪ {out['low']:,.0f} – ₪ {out['high']:,.0f}")
        else:
            self.lbl_result_range.configure(text="Typical range: —")

    def animate_value(self, current, target):
        diff = target - current
        step = diff / 10
        if abs(diff) < 10:
            self.lbl_result_val.configure(text=f"₪ {target:,.0f}")
        else:
            new_val = current + step
            self.lbl_result_val.configure(text=f"₪ {new_val:,.0f}")
            self.after(18, lambda: self.animate_value(new_val, target))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)


if __name__ == "__main__":
    app = SalaryCalculatorPro()
    app.mainloop()

