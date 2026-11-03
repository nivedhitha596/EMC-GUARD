# KONE EMC-GUARD – Deployment & Execution Guide

## Platform Overview
The **KONE EMC-GUARD** platform is a Physics-Informed Configuration-Aware EMC Digital Twin for elevator cable systems. It couples deterministic analytical physics formulations with high-fidelity machine learning surrogates and explainable AI to provide real-time capacitive and inductive coupling predictions, relative risk indicators, and Pareto-optimal configuration discovery.

---

## Prerequisites & System Requirements

- **Python Version:** Python 3.10 to 3.13 (Python 3.13.5 verified)
- **Supported Operating Systems:** Windows 10/11, Windows Server, Linux (Ubuntu 20.04+ / RHEL 8+), macOS (Apple Silicon / Intel)
- **RAM:** Minimum 2 GB (4 GB recommended)
- **Disk Space:** ~500 MB (including models, reports, and virtual environment)
- **Browser:** Chrome, Edge, Firefox, or Safari (Modern Evergreen browsers)

---

## 1. Quick Start (Local Execution)

### Step 1: Open Terminal / PowerShell at Project Root
```powershell
cd C:\Users\nived\OneDrive\Desktop\kone
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies: `streamlit>=1.30.0`, `plotly>=5.18.0`, `numpy>=1.26.0`, `pandas>=2.1.0`, `scipy>=1.11.0`, `scikit-learn>=1.3.0`, `joblib>=1.3.0`, `h5py>=3.9.0`, `reportlab>=4.0.0`)*

### Step 3: Launch the Dashboard
```bash
streamlit run EMC_PROJECT/app.py
```
Or with specific Python executable:
```powershell
& "D:\Python\python.exe" -m streamlit run EMC_PROJECT/app.py
```

The application will start on `http://localhost:8501`.

---

## 2. Headless / Server Deployment (Windows Service / Linux Daemon)

To run the Streamlit server continuously without local browser popping up:

### Command:
```bash
streamlit run EMC_PROJECT/app.py --server.headless true --server.port 8501 --server.address 0.0.0.0
```

### Streamlit Configuration (`.streamlit/config.toml`):
The repository contains the configuration in `EMC_PROJECT/.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#0055A5"
backgroundColor = "#070C18"
secondaryBackgroundColor = "#0C1629"
textColor = "#E2E8F0"
font = "sans serif"

[server]
headless = true
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 50
```

---

## 3. Docker Container Deployment

For containerised production deployments, use the following `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "EMC_PROJECT/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build & Run Container:
```bash
docker build -t kone-emc-guard:latest .
docker run -d -p 8501:8501 --name emc-guard kone-emc-guard:latest
```

---

## 4. Verification & Testing

Before deployment or after pulling updates, run the automated test suite:

```powershell
# Run Stage 3 & Stage 4 Comprehensive Unit Tests (11 test cases)
& "D:\Python\python.exe" AI_ML/tests/test_stage3_stage4.py

# Verify Compilation of Streamlit Application
& "D:\Python\python.exe" -c "import py_compile; py_compile.compile('EMC_PROJECT/app.py', doraise=True); print('OK')"
```

All 11 unit tests verify:
- Complete 216-configuration dataset evaluation without NaN or missing values.
- Correct mathematical normalisation and risk tier boundary classification ($[0,1]$ index).
- Multi-objective Pareto optimality correctness (strict mathematical dominance checking).
- Domain out-of-bounds rejection for invalid geometric parameters.
- Stage 1 and Stage 2 non-regression integrity.

---

## 5. Architectural & Scientific Principles

1. **Prediction Hierarchy:**
   - **Primary Engine:** Stage 12A Frozen MATLAB analytical formulation (`Stage12A_Selected_Models_FROZEN.mat`).
   - **Secondary Engine:** Stage 14A Extra Trees ensembles (`EMC_GUARD_Capacitance_ExtraTrees.joblib`, `EMC_GUARD_MutualInductance_ExtraTrees.joblib`).
2. **Scientific Integrity:**
   - Risk indicators ($I_C, I_M, I_{\text{comb}}$) are **relative metrics** computed strictly across the 216 discrete simulated configurations. They do **not** represent regulatory compliance (IEC/EN) or life-safety ratings.
   - Cross-validation results report both in-sample ($R^2=1.0$) and Leave-One-Level-Out (LOLO) metrics honestly, highlighting tree-based extrapolation bounds on unobserved physical boundary levels.
   - Pareto-optimal configurations (6 configurations) represent mathematical non-dominance within the discrete 216-point design grid.
3. **Data Immutability:**
   - All `.joblib` models, `.mat` files, and reference `.csv` datasets remain strictly frozen and uncorrupted.

