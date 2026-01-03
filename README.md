## Architect AI — Salary Estimator

This repo contains:

- **Modern GUI (recommended)**: `salary_app.py` (Streamlit)
- **Legacy desktop GUI**: `salary_gui_pro.py` (customtkinter)
- **CLI**: `salary_cli.py`

### Install

```bash
pip install -r requirements.txt
```

If you want the legacy desktop GUI too:

```bash
pip install -r requirements-desktop.txt
```

### Run (modern GUI)

```bash
streamlit run salary_app.py
```

Or:

```bash
python salary_gui.py
```

### Run (legacy desktop GUI)

```bash
python salary_gui_pro.py
```

### Run (CLI)

```bash
python salary_cli.py
```

### Deploy (easiest): Streamlit Community Cloud

- Push this repo to GitHub.
- Go to Streamlit Community Cloud and create a new app.
- Set:
  - **Main file path**: `salary_app.py`
  - **Python dependencies**: uses `requirements.txt` automatically
- Deploy.

### Deploy (portable): Docker (Render/Fly.io/any VM)

Build locally:

```bash
docker build -t seker-salary .
```

Run locally:

```bash
docker run --rm -p 8501:8501 seker-salary
```

Then open `http://localhost:8501`.



