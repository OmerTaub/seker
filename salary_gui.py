from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if importlib.util.find_spec("streamlit") is None:
        print("Streamlit is not installed.")
        print("Install dependencies with: pip install -r requirements.txt")
        raise SystemExit(1)

    app_path = Path(__file__).with_name("salary_app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print("Launching modern GUI:")
    print("  " + " ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import analyze_architecture_data as analysis
import threading

class SalaryCalculatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Israeli Architecture Salary Predictor")
        self.root.geometry("500x600")
        
        # Style
        style = ttk.Style()
        style.configure('TLabel', font=('Helvetica', 11))
        style.configure('TButton', font=('Helvetica', 11, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))
        
        # Status Label
        self.status_var = tk.StringVar()
        self.status_var.set("Loading data and training model... Please wait.")
        self.status_label = ttk.Label(root, textvariable=self.status_var, foreground="blue")
        self.status_label.pack(pady=10)
        
        # Main Frame (hidden until model loads)
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Variables
        self.gender_var = tk.StringVar(value="Female")
        self.age_var = tk.StringVar(value="26-30")
        self.inst_var = tk.StringVar(value="Technion")
        self.exp_var = tk.StringVar(value="1")
        self.size_var = tk.StringVar(value="11-30 Employees")
        self.loc_var = tk.StringVar(value="Center")
        self.degree_var = tk.StringVar(value="Bachelor")
        
        # Prediction Result
        self.result_var = tk.StringVar(value="--- NIS")
        
        # Start training in background
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            self.df = analysis.load_and_clean_data("data.csv")
            self.df = analysis.map_features(self.df)
            self.model, self.feature_cols = analysis.train_model(self.df)
            
            self.root.after(0, self.setup_ui)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {str(e)}"))

    def setup_ui(self):
        self.status_label.pack_forget()
        
        # Header
        ttk.Label(self.main_frame, text="Salary Calculator", style='Header.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Form Fields
        row = 1
        
        # Gender
        ttk.Label(self.main_frame, text="Gender:").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Combobox(self.main_frame, textvariable=self.gender_var, values=["Female", "Male"], state="readonly").grid(row=row, column=1, sticky='ew')
        row += 1
        
        # Age
        ttk.Label(self.main_frame, text="Age Group:").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Combobox(self.main_frame, textvariable=self.age_var, values=["20-25", "26-30", "31-35", "36-40", "41-45"], state="readonly").grid(row=row, column=1, sticky='ew')
        row += 1
        
        # Institution
        ttk.Label(self.main_frame, text="Institution:").grid(row=row, column=0, sticky='w', pady=5)
        insts = ["Technion", "Tel Aviv Univ", "Bezalel", "Ariel Univ", "Wizo Haifa", "Other"]
        ttk.Combobox(self.main_frame, textvariable=self.inst_var, values=insts, state="readonly").grid(row=row, column=1, sticky='ew')
        row += 1
        
        # Experience
        ttk.Label(self.main_frame, text="Experience (Years):").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Entry(self.main_frame, textvariable=self.exp_var).grid(row=row, column=1, sticky='ew')
        row += 1
        
        # Company Size
        ttk.Label(self.main_frame, text="Company Size:").grid(row=row, column=0, sticky='w', pady=5)
        sizes = ["1-10 Employees", "11-30 Employees", "31-50 Employees", "50+ Employees"]
        ttk.Combobox(self.main_frame, textvariable=self.size_var, values=sizes, state="readonly").grid(row=row, column=1, sticky='ew')
        row += 1
        
        # Location
        ttk.Label(self.main_frame, text="Location:").grid(row=row, column=0, sticky='w', pady=5)
        locs = ["Center", "Haifa", "Jerusalem", "North", "South", "Sharon"]
        ttk.Combobox(self.main_frame, textvariable=self.loc_var, values=locs, state="readonly").grid(row=row, column=1, sticky='ew')
        row += 1
        
        # Degree
        ttk.Label(self.main_frame, text="Degree Level:").grid(row=row, column=0, sticky='w', pady=5)
        ttk.Combobox(self.main_frame, textvariable=self.degree_var, values=["Bachelor", "Master"], state="readonly").grid(row=row, column=1, sticky='ew')
        row += 1
        
        # Separator
        ttk.Separator(self.main_frame, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=20)
        row += 1
        
        # Calculate Button
        calc_btn = ttk.Button(self.main_frame, text="Calculate Salary", command=self.calculate)
        calc_btn.grid(row=row, column=0, columnspan=2, pady=10, ipady=5)
        row += 1
        
        # Result Display
        result_frame = ttk.LabelFrame(self.main_frame, text="Estimated Monthly Salary", padding=15)
        result_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        
        ttk.Label(result_frame, textvariable=self.result_var, font=('Helvetica', 24, 'bold'), foreground="green", anchor="center").pack(fill=tk.X)
        
        # Grid config
        self.main_frame.columnconfigure(1, weight=1)

    def calculate(self):
        try:
            # Parse inputs
            size_map = {
                "1-10 Employees": 5,
                "11-30 Employees": 20,
                "31-50 Employees": 40,
                "50+ Employees": 60
            }
            
            exp_val = float(self.exp_var.get())
            
            user_profile = {
                'Gender_En': self.gender_var.get(),
                'Age': self.age_var.get(),
                'Institution_En': self.inst_var.get(),
                'Professional_Exp_Mapped': exp_val,
                'Company_Size_Mapped': size_map[self.size_var.get()],
                'Location_En': self.loc_var.get(),
                'Degree_Level': self.degree_var.get()
            }
            
            # Predict
            pred = analysis.predict_salary(self.model, self.feature_cols, user_profile)
            self.result_var.set(f"₪ {pred:,.0f}")
            
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure 'Experience' is a valid number.")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = SalaryCalculatorGUI(root)
    root.mainloop()

