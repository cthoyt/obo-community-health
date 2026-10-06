#!/usr/bin/env -S uv run --script

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "bioregistry>=0.15.0",
#     "pandas>=3.0.6",
#     "ratelimit>=2.2.1",
# ]
# ///

"""Build an ODK summary file."""

from itertools import islice
from pathlib import Path

import bioregistry
import click
import pandas as pd
import requests
import yaml
from pydantic import BaseModel
from pystow.github import search_code
from tqdm import tqdm

from utils import ODK_REPOS_YAML_PATH


class Row(BaseModel):
    repository: str
    name: str
    path: str
    version: str
    prefix: str | None = None


COLUMNS = ["repository", "name", "path", "version", "prefix"]

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

QUERY = (f"filename:odk.yaml {SKIP_Q} -is:fork",)


def get_repository_to_bioregistry() -> dict[str, str]:
    rv = {}
    for resource in bioregistry.resources():
        repository = resource.get_repository()
        if not repository or not repository.startswith("https://github.com/"):
            continue
        rv[repository.removeprefix("https://github.com/").casefold()] = resource.prefix
    return rv


@click.command()
@click.option("--per-page", type=int, default=40)
@click.option("--output-path", default=ODK_REPOS_YAML_PATH, type=Path)
@click.option("--refresh", is_flag=True)
def main(per_page: int, output_path: Path, refresh: bool) -> None:
    data: dict[str, Row]
    if output_path.is_file() and not refresh:
        data = {
            record["repository"]: Row.model_validate(record)
            for record in yaml.safe_load(output_path.read_text())
        }
    else:
        data = {}

    repository_to_bioregistry = get_repository_to_bioregistry()

    for item in search_code(QUERY, page_size=per_page):
        path = item["path"]
        if not path.endswith(".yaml"):
            continue

        name = item["name"]

        # branch = get_default_branch()
        branch = "master"  # FIXME deal with this, since is main on newer repos
        repository = item["repository"]["full_name"]
        url = f"https://raw.githubusercontent.com/{repository}/{branch}/src/ontology/Makefile"
        try:
            line, *_ = islice(
                requests.get(url, stream=True, timeout=15).iter_lines(decode_unicode=True), 3, 4
            )
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

    model_rows = sorted(data.values(), key=lambda row: row.repository.casefold())
    rows = [m.dict(exclude_none=True) for m in model_rows]

    df = pd.DataFrame(rows)
    df = df[COLUMNS]
    df.to_csv(output_path.with_suffix(".tsv"), sep="\t", index=False)

    output_path.write_text(yaml.safe_dump(rows))


if __name__ == "__main__":
    main()
