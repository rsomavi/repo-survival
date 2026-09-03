# OSS Pulse

Predicts whether a GitHub repository will still be active months or years
after creation, using only signals from its first 30 days — commits,
contributors, and how issues were handled.

Built for the DATA11001 Data Science course at the University of Helsinki.

## How it works

1. **Data collection** — repositories are sampled from the GitHub API,
   stratified by creation quarter and star range to avoid bias toward
   currently-active repos.
2. **Feature engineering** — commit frequency, contributor count, issue
   close ratio, and repo metadata are computed for each repo's first month.
3. **Model** — a Random Forest trained on 2,397 Python repositories
   (AUC 0.817) predicts survival probability.
4. **Web app** — paste any GitHub URL and get a live prediction, computed
   from that repo's actual first-month history.

## Project structure

```
repo-survival/
├── data/
│   ├── raw/              # raw API responses (gitignored)
│   └── processed/        # cleaned dataset (repos_clean.csv)
├── notebooks/
│   └── 01_eda.ipynb      # exploratory analysis + model training
├── src/
│   ├── collect_search.py   # stratified repo sampling
│   ├── collect_details.py  # per-repo feature collection
│   ├── clean.py             # cleaning + feature engineering
│   ├── model/                # trained model (.pkl)
│   └── api/
│       └── app.py            # Flask backend
├── web/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── requirements.txt
```

## Running it locally

**1. Set up the environment**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Get a GitHub token**

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. No scopes are needed — read access to public repos works without any
4. Copy the token and add it to a `.env` file in the project root:

```
GITHUB_TOKEN=your_token_here
```

**3. Run the backend**

```bash
python src/api/app.py
```

Runs on `http://localhost:5000`.

**4. Serve the frontend**

In a separate terminal:

```bash
cd web
python3 -m http.server 8000
```

Open `http://localhost:8000` in your browser.

**5. (Optional) Reproduce the data pipeline**

The cleaned dataset is already included in `data/processed/`, so this
step isn't required to run the app. To regenerate it from scratch:

```bash
python src/collect_search.py
python src/collect_details.py
python src/clean.py
```

## Known limitations

- Trained on Python repositories created in 2023–2024 only.
- Solo or academic projects that don't use GitHub Issues or a public
  license tend to score as "likely abandoned" even when actively worked
  on, since those are the same signals unmaintained repos show.
- "Active" means pushed to recently — not popular, not well-maintained.

## License

MIT
