"""Build an ODK summary file."""

import json
from itertools import islice
from pathlib import Path
from typing import TypedDict

import bioregistry
import click
import pandas as pd
import requests
import yaml
from tqdm import tqdm

from utils import ODK_REPOS_YAML_PATH, get_github


class Row(TypedDict):
    repository: str
    name: str
    path: str
    version: str
    prefix: str | None


COLUMNS = ["repository", "name", "path", "version", "prefix"]


@click.command()
@click.option("--per-page", type=int, default=40)
@click.option("--path", default=ODK_REPOS_YAML_PATH, type=Path)
def main(per_page: int, path: Path):
    data: dict[str, Row]
    if path.is_file():
        data = {record["repository"]: record for record in yaml.safe_load(path.read_text())}
    else:
        data = {}

    repository_to_bioregistry = get_repository_to_bioregistry()

    page = 1
    tqdm.write(f"loading page {page} of size {per_page}")
    total = paginate_github_search(
        data, per_page, page, repository_to_bioregistry=repository_to_bioregistry
    )
    tqdm.write(f"after page {page}, found that there are {total} rows.")

    while per_page * page < total:
        page += 1
        tqdm.write(f"loading page {page} of size {per_page}")
        paginate_github_search(
            data, per_page, page, repository_to_bioregistry=repository_to_bioregistry
        )

    rows = sorted(data.values(), key=lambda row: row["repository"].casefold())

    tsv_path = path.with_suffix(".tsv")

    df = pd.DataFrame(rows)
    df = df[COLUMNS]
    df.to_csv(tsv_path, sep="\t", index=False)

    path.write_text(yaml.safe_dump(rows))


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
    "OBOAcademy",  # teaching material
]

#: Build the GitHub query for skipping certain users
SKIP_Q = " ".join(f"-user:{user}" for user in SKIP_USERS)


def paginate_github_search(
    data: dict[str, Row],
    per_page: int,
    page: int,
    repository_to_bioregistry: dict[str, str],
) -> int:
    res = get_github(
        "https://api.github.com/search/code",
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
            line, *_ = islice(requests.get(url, stream=True).iter_lines(decode_unicode=True), 3, 4)
            version = line.removeprefix("# ODK Version: v")
        except ValueError:
            tqdm.write(f"Could not get ODK version in {path} in {repository}")
            version = "unknown"
        data[repository] = Row(
            repository=repository,
            name=name,
            version=version,
            path=path,
            prefix=repository_to_bioregistry.get(repository.casefold()),
        )
    return res["total_count"]


def get_repository_to_bioregistry() -> dict[str, str]:
    rv = {}
    for resource in bioregistry.resources():
        repository = resource.get_repository()
        if not repository or not repository.startswith("https://github.com/"):
            continue
        rv[repository.removeprefix("https://github.com/").casefold()] = resource.prefix
    return rv


if __name__ == "__main__":
    main()
