import analyze_architecture_data as analysis
import time

def get_choice(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")
    
    while True:
        try:
            choice = input("Enter number: ")
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid choice, please try again.")

def main():
    print("="*50)
    print("   ARCHITECT SALARY CALCULATOR (TERMINAL)   ")
    print("="*50)
    print("Loading data and training model... please wait...")
    
    # Load cached model if available (or train + cache on first run)
    bundle = analysis.get_or_train_model(data_path="data.csv", cache_path="salary_model.joblib", force_retrain=False, verbose=True)
    
    print("Model ready!\n")
    
    while True:
        print("\n--- Enter Profile Details ---")
        
        gender = get_choice("Gender:", ["Female", "Male"])
        age = get_choice("Age Group:", ["20-25", "26-30", "31-35", "36-40", "41-45"])
        inst = get_choice("Institution:", ["Technion", "Tel Aviv Univ", "Bezalel", "Ariel Univ", "Wizo Haifa", "Other"])
        
        while True:
            try:
                exp = float(input("\nYears of Experience (e.g., 0, 1.5, 5): "))
                break
            except ValueError:
                print("Please enter a valid number.")
                
        size_label = get_choice("Company Size:", ["1-10 Employees", "11-30 Employees", "31-50 Employees", "50+ Employees"])
        size_map = {
            "1-10 Employees": 5,
            "11-30 Employees": 20,
            "31-50 Employees": 40,
            "50+ Employees": 60
        }
        
        loc = get_choice("Location:", ["Center", "Haifa", "Jerusalem", "North", "South", "Sharon"])
        degree = get_choice("Degree Level:", ["Bachelor", "Master"])

        work_mode = get_choice("Work Mode:", ["On-site", "Hybrid", "Remote"])
        overtime = get_choice("Overtime Frequency:", ["Never", "Rarely", "Monthly", "Weekly"])

        while True:
            try:
                daily_hours = float(input("\nDaily work hours (e.g., 8 or 9): "))
                break
            except ValueError:
                print("Please enter a valid number.")

        while True:
            try:
                weekly_days = float(input("Work days per week (e.g., 5): "))
                break
            except ValueError:
                print("Please enter a valid number.")

        keren = get_choice("Keren Hishtalmut:", ["No", "Yes"])
        meals = get_choice("Meals (Cibus/TenBis):", ["No", "Yes"])

        # Multi-field selection
        field_options = [
            ("Residential", "Field_Residential"),
            ("Public Buildings", "Field_Public"),
            ("TAMA 38 / Urban Renewal", "Field_TAMA38"),
            ("Interior Design", "Field_Interior"),
            ("Commercial / Offices", "Field_Commercial"),
            ("Project Management", "Field_ProjectManagement"),
            ("Landscape", "Field_Landscape"),
        ]
        print("\nField(s) (comma-separated numbers, or Enter for General/Other):")
        for i, (label, _) in enumerate(field_options, 1):
            print(f"{i}. {label}")
        raw = input("Choose: ").strip()
        field_features = {k: 0 for _, k in field_options}
        if raw:
            try:
                picked = {int(x.strip()) for x in raw.split(",") if x.strip()}
                for idx in picked:
                    if 1 <= idx <= len(field_options):
                        field_features[field_options[idx - 1][1]] = 1
            except ValueError:
                print("Invalid field selection; defaulting to General/Other.")
        
        user_profile = {
            'Gender_En': gender,
            'Age': age,
            'Institution_En': inst,
            'Professional_Exp_Mapped': exp,
            'Company_Size_Mapped': size_map[size_label],
            'Location_En': loc,
            'Degree_Level': degree,
            'Work_Mode_En': work_mode,
            'Overtime_Freq_En': overtime,
            'Daily_Hours': daily_hours,
            'Weekly_Days': weekly_days,
            'Keren_Hishtalmut_En': keren,
            'Meals_En': meals,
            **field_features,
        }
        
        print("\nCalculating...")
        time.sleep(0.5)
        out = analysis.predict_salary_with_range(bundle, user_profile, band="p80")
        pred = out["pred"]
        
        print("-" * 30)
        print(f"ESTIMATED SALARY: ₪ {pred:,.0f} / month")
        if "low" in out and "high" in out:
            print(f"Typical range (80%): ₪ {out['low']:,.0f} – ₪ {out['high']:,.0f}")
        print("-" * 30)
        
        if input("\nCalculate another? (y/n): ").lower() != 'y':
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    main()

