"""Build an ODK summary file."""

import json
from itertools import islice

import click
import pandas as pd
import requests
from tqdm import tqdm

from utils import ODK_REPOS_PATH, get_github


@click.command()
def main():
    per_page = 100
    page = 1
    rows = set()
    total = _xxx(rows, per_page, page)
    click.echo(f"Got {total} rows")
    while per_page * page < total:
        page += 1
        _xxx(rows, per_page, page)

    df = pd.DataFrame(sorted(rows), columns=["repository", "name", "version"])
    df.to_csv(ODK_REPOS_PATH, sep="\t", index=False)


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
            version = "unknown"
        rows.add((repository, name, version))
    return res["total_count"]


if __name__ == "__main__":
    main()
