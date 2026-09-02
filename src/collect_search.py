"""
Busca repos de GitHub creados entre 2023-2024, estratificando por
trimestre de creación y rango de stars, para evitar sesgo hacia
repos con actividad reciente o solo hacia los muy populares.
Usa /search/repositories (límite de 30 peticiones/hora).
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise ValueError("GITHUB_TOKEN no encontrado. Revisa tu archivo .env")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}

SEARCH_URL = "https://api.github.com/search/repositories"
LANGUAGE = "python"
OUTPUT_PATH = "data/raw/repos_list.json"

# 8 trimestres cubriendo 2023 y 2024
QUARTERS = [
    ("2023-01-01", "2023-03-31"),
    ("2023-04-01", "2023-06-30"),
    ("2023-07-01", "2023-09-30"),
    ("2023-10-01", "2023-12-31"),
    ("2024-01-01", "2024-03-31"),
    ("2024-04-01", "2024-06-30"),
    ("2024-07-01", "2024-09-30"),
    ("2024-10-01", "2024-12-31"),
]

# Rangos de stars: sin tracción, tracción media, tracción alta
STAR_RANGES = [
    "0",
    "1..49",
    "50..*",
]


def search_query(date_start, date_end, stars_range, per_page=100):
    """Una sola query a /search/repositories, sin sort (evita sesgo)."""
    params = {
        "q": f"created:{date_start}..{date_end} language:{LANGUAGE} stars:{stars_range}",
        "per_page": per_page,
    }

    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)

    if response.status_code != 200:
        print(f"  Error: {response.status_code} - {response.text[:200]}")
        return []

    return response.json().get("items", [])


def collect_stratified():
    all_repos = []
    query_count = 0

    for date_start, date_end in QUARTERS:
        for stars_range in STAR_RANGES:
            query_count += 1
            print(f"[{query_count}/{len(QUARTERS) * len(STAR_RANGES)}] "
                  f"{date_start}..{date_end}, stars:{stars_range}")

            items = search_query(date_start, date_end, stars_range)
            all_repos.extend(items)
            print(f"  -> {len(items)} repos")

            # Margen de cortesía sobre el límite de 30/hora
            # (30 peticiones/hora = 1 cada 2 min como máximo teórico,
            # pero GitHub reparte el cupo por ventana deslizante, así
            # que con 24 queries y una pausa moderada suele ir bien)
            time.sleep(5)

    return all_repos


def save_repos(repos, path):
    # Deduplicar por id, ya que un repo puede colar en más de un rango
    # si sus stars cambiaron de categoría entre queries (poco probable
    # pero posible), o por solapamiento de resultados
    seen = {}
    for repo in repos:
        seen[repo["id"]] = repo

    unique_repos = list(seen.values())

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(unique_repos, f, indent=2, ensure_ascii=False)

    print(f"\nTotal bruto: {len(repos)} | Únicos tras deduplicar: {len(unique_repos)}")
    print(f"Guardado en {path}")


if __name__ == "__main__":
    repos = collect_stratified()
    save_repos(repos, OUTPUT_PATH)