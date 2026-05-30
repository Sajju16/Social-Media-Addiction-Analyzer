# Social Media Addiction Analyzer (Streamlit)

Dashboard for:
- Screen time analysis (optional CSV upload)
- Addiction score (0-100)
- Charts (matplotlib)
- ML risk prediction (scikit-learn)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run app.py
```

## CSV Format (optional)

The app tries to extract daily total hours if your CSV contains a `date` column.

Supported shapes:
- `date, platform, hours`  (long format)
- `date, hours` (daily rows)
- `date, social_hours`

Platform breakdown needs a `platform` + `hours` (or `platform` + `social_hours`) pair.

## Notes

- The tool is for awareness and self-reflection, not a medical diagnosis.
- ML model is trained on synthetic data generated from the heuristic rules used for the addiction score.

