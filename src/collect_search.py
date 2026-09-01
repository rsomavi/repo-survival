"""
Busca repos de GitHub creados en un rango de fechas y guarda la lista
en bruto para procesarla después. Usa el endpoint /search/repositories,
que tiene un límite de 30 peticiones/hora.
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

# Rango de fechas: repos creados entre 2023 y 2024
DATE_RANGE = "2023-01-01..2024-12-31"
LANGUAGE = "python"  # empezamos con un solo lenguaje para tener una muestra homogénea

OUTPUT_PATH = "data/raw/repos_list.json"


def search_repos(max_pages=10, per_page=100):
    """
    Pagina por los resultados de búsqueda. GitHub limita a un máximo
    de 1000 resultados por query (page * per_page <= 1000).
    """
    all_repos = []

    for page in range(1, max_pages + 1):
        params = {
            "q": f"created:{DATE_RANGE} language:{LANGUAGE}",
            "sort": "updated",  # evita que salgan solo los más populares
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }

        response = requests.get(SEARCH_URL, headers=HEADERS, params=params)

        if response.status_code != 200:
            print(f"Error en página {page}: {response.status_code} - {response.text}")
            break

        data = response.json()
        items = data.get("items", [])

        if not items:
            print(f"No hay más resultados en la página {page}. Parando.")
            break

        all_repos.extend(items)
        print(f"Página {page}: {len(items)} repos recogidos (total: {len(all_repos)})")

        # Respetar el rate limit de búsqueda (30/hora ≈ 1 cada 2 segundos mínimo,
        # pero mejor ir sobrados)
        time.sleep(2)

    return all_repos


def save_repos(repos, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(repos, f, indent=2, ensure_ascii=False)
    print(f"\nGuardados {len(repos)} repos en {path}")


if __name__ == "__main__":
    repos = search_repos(max_pages=10, per_page=100)
    save_repos(repos, OUTPUT_PATH)
