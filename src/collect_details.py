"""
Por cada repo de data/raw/repos_list.json, saca features detalladas:
contribuidores, actividad de commits en el primer mes, e issues.
Usa endpoints con límite de 5000/hora, así que hay más margen que
en la búsqueda, pero con muchos repos x varias llamadas cada uno,
puede tardar un rato — incluye pausas y guarda progreso poco a poco
por si se corta a mitad.
"""

import os
import json
import time
import argparse
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}

INPUT_PATH = "data/raw/repos_list.json"
OUTPUT_PATH = "data/raw/repos_details.json"


def get_paginated_count(url, params=None):
    """
    Trucazo para contar elementos sin descargarlos todos: pide 1 item
    por página y mira en la cabecera 'Link' cuál es la última página.
    Si no hay cabecera Link, es que hay 0 o 1 elementos.
    """
    params = params or {}
    params["per_page"] = 1
    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code != 200:
        return 0

    if "Link" not in response.headers:
        return len(response.json())

    links = response.headers["Link"]
    for part in links.split(","):
        if 'rel="last"' in part:
            last_page_url = part.split(";")[0].strip("<> ")
            last_page = int(last_page_url.split("page=")[-1].split("&")[0])
            return last_page

    return len(response.json())


def get_first_month_commits(owner, repo, created_at):
    """Cuenta commits en los primeros 30 días de vida del repo."""
    created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    until = created + timedelta(days=30)

    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {
        "since": created.isoformat() + "Z",
        "until": until.isoformat() + "Z",
    }
    return get_paginated_count(url, params)


def get_contributors_count(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
    return get_paginated_count(url, {"anon": "false"})


GRAPHQL_URL = "https://api.github.com/graphql"

def get_issues_counts(owner, repo):
    """Cuenta issues abiertas y cerradas usando GraphQL (conteo exacto)."""
    query = """
    query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) {
        openIssues: issues(states: OPEN) { totalCount }
        closedIssues: issues(states: CLOSED) { totalCount }
      }
    }
    """
    variables = {"owner": owner, "name": repo}
    response = requests.post(
        GRAPHQL_URL,
        headers=HEADERS,
        json={"query": query, "variables": variables},
    )

    if response.status_code != 200:
        return 0, 0

    data = response.json().get("data", {}).get("repository")
    if not data:
        return 0, 0

    return data["openIssues"]["totalCount"], data["closedIssues"]["totalCount"]


def collect_details(repos, output_path, checkpoint_every=25):
    results = []

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        repos = [r for r in repos if r["id"] not in done_ids]
        print(f"Retomando: ya había {len(results)} repos procesados, quedan {len(repos)}")

    for i, repo in enumerate(repos):
        owner = repo["owner"]["login"]
        name = repo["name"]

        try:
            first_month_commits = get_first_month_commits(owner, name, repo["created_at"])
            contributors = get_contributors_count(owner, name)
            open_issues, closed_issues = get_issues_counts(owner, name)

            results.append({
                "id": repo["id"],
                "full_name": repo["full_name"],
                "created_at": repo["created_at"],
                "pushed_at": repo["pushed_at"],
                "stargazers_count": repo["stargazers_count"],
                "forks_count": repo["forks_count"],
                "has_license": repo["license"] is not None,
                "has_wiki": repo["has_wiki"],
                "has_description": repo["description"] is not None,
                "first_month_commits": first_month_commits,
                "contributors_count": contributors,
                "open_issues": open_issues,
                "closed_issues": closed_issues,
            })

            print(f"[{i+1}/{len(repos)}] {repo['full_name']} — OK")

        except Exception as e:
            print(f"[{i+1}/{len(repos)}] {repo['full_name']} — ERROR: {e}")

        if (i + 1) % checkpoint_every == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"--- Checkpoint guardado ({len(results)} repos) ---")

        time.sleep(0.3)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nCompletado: {len(results)} repos guardados en {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recoge detalles de repos de GitHub")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Número de repos a procesar (por defecto: todos)"
    )
    args = parser.parse_args()

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        all_repos = json.load(f)

    repos = all_repos[:args.limit] if args.limit else all_repos

    if args.limit:
        print(f"Modo limitado: procesando {len(repos)} de {len(all_repos)} repos totales")

    collect_details(repos, OUTPUT_PATH)