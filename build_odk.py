"""Build an ODK summary file."""

import json
from itertools import islice

import requests
import yaml
from tqdm import tqdm

from utils import ODK_REPOS_PATH, get_github


def main():
    per_page = 100
    page = 1
    rows = set()
    total = _xxx(rows, per_page, page)
    while per_page * page < total:
        page += 1
        _xxx(rows, per_page, page)

    with ODK_REPOS_PATH.open("w") as file:
        yaml.safe_dump(
            [dict(zip(("repository", "name", "version"), row)) for row in sorted(rows)],
            file,
        )


def _xxx(rows, per_page, page):
    res = get_github(
        f"https://api.github.com/search/code",
        params={
            "per_page": per_page,
            "page": page,
            # "sort": "indexed",
            "q": "filename:odk.yaml -user:INCATools -user:matentzn -user:one-acre-fund",
        },
    )
    if "items" not in res:
        raise KeyError("\n" + json.dumps(res, indent=2))
    for item in tqdm(res["items"]):
        name = item["name"]
        repository = item["repository"]["full_name"]
        url = f"https://raw.githubusercontent.com/{repository}/master/src/ontology/Makefile"
        try:
            line, *_ = islice(
                requests.get(url, stream=True).iter_lines(decode_unicode=True), 3, 4
            )
            version = line.removeprefix("# ODK Version: v")
        except ValueError:
            tqdm.write(f"Could not get ODK version for {name} in {repository}")
            version = None
        rows.add((repository, name, version))
    return res["total_count"]


if __name__ == "__main__":
    main()
