"""Build an ODK summary file."""

import json
from itertools import islice
from typing import TypedDict

import click
import pandas as pd
import requests
import yaml
from tqdm import tqdm

from utils import ODK_REPOS_PATH, ODK_REPOS_YAML_PATH, get_github


class Row(TypedDict):
    repository: str
    name: str
    path: str
    version: str


COLUMNS = ["repository", "name", "path", "version"]


@click.command()
@click.option("--per-page", type=int, default=40)
def main(per_page: int):
    data: dict[str, Row] = {
        record["repository"]: record
        for record in yaml.safe_load(ODK_REPOS_YAML_PATH.read_text())
    }

    page = 1
    tqdm.write(f"loading page {page} of size {per_page}")
    total = _paginate(data, per_page, page)
    tqdm.write(f"after page {page}, found that there are {total} rows.")

    while per_page * page < total:
        page += 1
        tqdm.write(f"loading page {page} of size {per_page}")
        _paginate(data, per_page, page)


    _rows = sorted(data.values(), key=lambda row: row["repository"].casefold())

    df = pd.DataFrame(_rows)
    df = df[COLUMNS]
    df.to_csv(ODK_REPOS_PATH, sep="\t", index=False)

    ODK_REPOS_YAML_PATH.write_text(yaml.safe_dump(_rows))


#: Users who have many test ODK files
#: or other reasons to not be considered
SKIP_USERS = [
    "INCATools",
    "matentzn",
    "one-acre-fund",
    "agustincharry",  # kafka stuff
    "hboutemy",  # hboutemy/mcmm-yaml is not related
    "kirana-ks",  # kirana-ks/aether-infrastructure-provisioning is not related to ODK
    "ferjavrec",  # projects in odk-central are not related to our ODK
    "acevesp",
    "cthoyt",  # self reference
]

#: Build the GitHub query for skipping certain users
SKIP_Q = " ".join(f"-user:{user}" for user in SKIP_USERS)


def _paginate(data: dict[str, Row], per_page: int, page: int) -> int:
    res = get_github(
        f"https://api.github.com/search/code",
        params={
            "per_page": per_page,
            "page": page,
            "sort": "indexed",
            "q": f"filename:odk.yaml {SKIP_Q}",
        },
    )
    if "items" not in res:
        raise KeyError("\n" + json.dumps(res, indent=2))
    for item in tqdm(res["items"], desc=f"Page {page}"):
        name = item["name"]
        path = item["path"]
        branch = "master"  # FIXME deal with this, since is main on newer repos
        repository = item["repository"]["full_name"]
        url = f"https://raw.githubusercontent.com/{repository}/{branch}/src/ontology/Makefile"
        try:
            line, *_ = islice(
                requests.get(url, stream=True).iter_lines(decode_unicode=True), 3, 4
            )
            version = line.removeprefix("# ODK Version: v")
        except ValueError:
            tqdm.write(f"Could not get ODK version in {path} in {repository}")
            version = "unknown"
        data[repository] = Row(
            repository=repository, name=name, version=version, path=path
        )
    return res["total_count"]


if __name__ == "__main__":
    main()
