"""Build an ODK summary file."""

import json

import yaml

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
        yaml.safe_dump(dict(rows), file)


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
    for item in res["items"]:
        rows.add((item["repository"]["full_name"], item["name"]))
    return res["total_count"]


if __name__ == "__main__":
    main()
