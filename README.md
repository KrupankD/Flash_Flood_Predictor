# Haor Flash Flood Prediction System

Predicts whether a flash flood is occurring in the haor wetlands (Sunamganj/Sylhet, NE Bangladesh)
from satellite and reanalysis variables. Logistic Regression + Random Forest, Streamlit UI.

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
# PowerShell blocked? run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
```

## Run
```bash
python train.py                # trains, cross-validates, saves models/
python -m streamlit run app.py           # opens the web UI at http://localhost:8501
```

## Files
| File | Purpose |
|---|---|
| `data/haor_flood_events_v1.csv` | 77 labelled events (Mendeley Data) |
| `common.py` | paths, feature list, cleaning, risk thresholds |
| `train.py` | trains Logistic Regression and Random Forest, repeated CV, saves artifacts |
| `app.py` | Streamlit UI: Predict, Model performance, Data explorer, About |
| `models/` | created by `train.py` (model, metrics, importances) |

## Key design decisions
- `source` column dropped: it leaks the label.
- `slope`, `twi` dropped: constant across the study area.
- 5-fold stratified CV repeated 10 times, because 77 rows make a single split unreliable.
- Random Forest depth limited (max_depth=5, min_samples_leaf=2) to limit overfitting.
- Missing NDWI (5 events) filled with the median inside each pipeline.

## Limitations
Small single-region dataset; nowcasting (not forecasting); educational use only.
