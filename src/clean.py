"""
Limpia los datos en bruto de data/raw/repos_details.json:
- elimina duplicados
- calcula features derivadas
- define la variable objetivo (éxito/abandono)
- guarda el resultado en data/processed/repos_clean.csv
"""

import json
import pandas as pd
from datetime import datetime, timezone

INPUT_PATH = "data/raw/repos_details.json"
OUTPUT_PATH = "data/processed/repos_clean.csv"

ACTIVE_THRESHOLD_DAYS = 180  # 6 meses
STARS_THRESHOLD = 10
CONTRIBUTORS_THRESHOLD = 3


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(f"Cargados {len(df)} registros")
    return df


def deduplicate(df):
    before = len(df)
    df = df.drop_duplicates(subset="id", keep="first")
    print(f"Duplicados eliminados: {before - len(df)}")
    return df


def parse_dates(df):
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["pushed_at"] = pd.to_datetime(df["pushed_at"])
    return df


def add_features(df):
    now = pd.Timestamp.now(tz="UTC")

    # Edad del repo en días
    df["age_days"] = (now - df["created_at"]).dt.days

    # Días desde el último push
    df["days_since_push"] = (now - df["pushed_at"]).dt.days

    # Ratio de issues cerradas (evitando división por cero)
    total_issues = df["open_issues"] + df["closed_issues"]
    df["issue_close_ratio"] = df["closed_issues"] / total_issues.replace(0, pd.NA)
    df["issue_close_ratio"] = df["issue_close_ratio"].fillna(0)

    # Commits por contribuidor en el primer mes (evitando división por cero)
    df["commits_per_contributor"] = df["first_month_commits"] / df["contributors_count"].replace(0, pd.NA)
    df["commits_per_contributor"] = df["commits_per_contributor"].fillna(0)

    return df


def add_label(df):
    is_active = df["days_since_push"] <= ACTIVE_THRESHOLD_DAYS
    has_traction = (df["stargazers_count"] >= STARS_THRESHOLD) | (df["contributors_count"] >= CONTRIBUTORS_THRESHOLD)

    # Éxito = activo (con o sin tracción). Abandonado = sin actividad reciente.
    df["label"] = is_active.astype(int)
    df["has_traction"] = has_traction.astype(int)

    print(f"\nDistribución de la variable objetivo:")
    print(df["label"].value_counts())
    print(f"\nDe los activos, con tracción: {(is_active & has_traction).sum()}")

    return df


def clean(input_path, output_path):
    df = load_data(input_path)
    df = deduplicate(df)
    df = parse_dates(df)
    df = add_features(df)
    df = add_label(df)

    df.to_csv(output_path, index=False)
    print(f"\nGuardado: {output_path} ({len(df)} filas, {len(df.columns)} columnas)")

    return df


if __name__ == "__main__":
    clean(INPUT_PATH, OUTPUT_PATH)
