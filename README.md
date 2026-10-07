# OSS Pulse

Predicts whether a GitHub repository will still be active months or years
after creation, using only signals from its first 30 days — commits,
contributors, and how issues were handled.

**Live demo: https://repo-survival.onrender.com/** — the free hosting plan
sleeps when idle, so the first load can take up to a minute.

Built for the DATA11001 Data Science course at the University of Helsinki.
The full write-up, with the data analysis, figures and model comparison, is
in [`docs/report.md`](docs/report.md).

## How it works

1. **Data collection** — repositories are sampled from the GitHub API,
   stratified by creation quarter and star range to avoid bias toward
   currently-active repos.
2. **Feature engineering** — commit frequency, contributor count, issue
   close ratio, and repo metadata are computed for each repo's first month.
3. **Model** — four model families (logistic regression, Random Forest,
   Gradient Boosting, MLP) were compared and tuned with cross-validation on
   2,397 Python repositories. The final model is a Gradient Boosting
   classifier (test AUC 0.826).
4. **Web app** — paste any GitHub URL and get a live prediction, computed
   from that repo's actual first-month history.

## Project structure

```
repo-survival/
├── data/
│   ├── raw/                  # raw API responses (gitignored)
│   └── processed/            # cleaned dataset (repos_clean.csv)
├── docs/
│   ├── report.md             # project report
│   └── images/               # figures used in the report
├── notebooks/
│   ├── 01_eda.ipynb          # exploratory analysis
│   └── 02_modeling.ipynb     # model comparison, tuning, export
├── src/
│   ├── collect_search.py     # stratified repo sampling
│   ├── collect_details.py    # per-repo feature collection
│   ├── clean.py              # cleaning + feature engineering
│   ├── model/                # trained model (.pkl)
│   └── api/
│       └── app.py            # Flask backend (also serves the frontend)
├── tests/                    # pytest suite
├── web/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── data/stats.json       # stats shown on the page
├── .env.example
├── requirements.txt          # runtime dependencies (used by the deployment)
└── requirements-dev.txt      # notebooks and tests
```

## Running it locally

**1. Set up the environment**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

Use `requirements.txt` instead if you only want to run the app, without the
notebooks and tests.

**2. Get a GitHub token**

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. No scopes are needed — read access to public repos works without any
4. Copy the example file and put the token in it:

```bash
cp .env.example .env
```

```
GITHUB_TOKEN=your_token_here
```

**3. Run the app**

```bash
python src/api/app.py
```

Open `http://localhost:5000`. Flask serves both the API and the web page.

**4. (Optional) Reproduce the data pipeline**

The cleaned dataset is already included in `data/processed/`, so this
step isn't required to run the app. To regenerate it from scratch:

```bash
python src/collect_search.py
python src/collect_details.py
python src/clean.py
```

The notebooks in `notebooks/` read the cleaned dataset. Run `01_eda.ipynb`
first and then `02_modeling.ipynb` (Kernel → Restart & Run All); the second
one retrains the model and rewrites `src/model/*.pkl` and
`web/data/stats.json`.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The API tests mock GitHub and the model, so they run offline and do not use
any API quota. The data and model tests use the files in `data/processed/`
and `src/model/`.

## Deployment

The app runs as a single web service on [Render](https://render.com)
(Python 3.10):

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn --chdir src/api app:app --workers 1 --threads 4 --timeout 120`
- **Environment variables:** `GITHUB_TOKEN` and `PYTHON_VERSION=3.10.12`
- **Health check path:** `/health`

Library versions in `requirements.txt` are pinned to the ones used for
training, because a saved scikit-learn model is not guaranteed to load with a
different version.

## Known limitations

- Trained on Python repositories created in 2023–2024 only.
- Solo or academic projects that don't use GitHub Issues or a public
  license tend to score as "likely abandoned" even when actively worked
  on, since those are the same signals unmaintained repos show.
- "Active" means pushed to recently — not popular, not well-maintained.
- The live demo shares one GitHub token, so under heavy use it can hit the
  GitHub API rate limit.

## License

MIT