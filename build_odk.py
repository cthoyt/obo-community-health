"""Build an ODK summary file."""

import json
from itertools import islice
from typing import NamedTuple

import click
import pandas as pd
import requests
from tqdm import tqdm

from utils import ODK_REPOS_PATH, get_github


class Row(NamedTuple):
    repository: str
    name: str
    version: str


@click.command()
def main():
    per_page = 100
    page = 1
    rows: set[Row] = set()
    total = _xxx(rows, per_page, page)
    click.echo(f"Got {total} rows. Already put in {len(rows)}")
    while per_page * page < total:
        page += 1
        _xxx(rows, per_page, page)

    df = pd.DataFrame(sorted(rows), columns=["repository", "name", "version"])
    df.to_csv(ODK_REPOS_PATH, sep="\t", index=False)


#: Users who have many test ODK files
#: or other reasons to not be considered
SKIP_USERS = [
    "INCATools",
    "matentzn",
    "one-acre-fund",
]

#: Build the GitHub query for skipping certain users
SKIP_Q = " ".join(f"-user:{user}" for user in SKIP_USERS)


def _xxx(rows: set[Row], per_page: int, page: int) -> int:
    res = get_github(
        f"https://api.github.com/search/code",
        params={
            "per_page": per_page,
            "page": page,
            # "sort": "indexed",
            "q": f"filename:odk.yaml {SKIP_Q}",
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
        rows.add(Row(repository=repository, name=name, version=version))
    return res["total_count"]


if __name__ == "__main__":
    main()
