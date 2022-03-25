"""Build the contacts data table."""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Optional

import click
import dateparser
import pandas as pd
import yaml
from tqdm import tqdm

from utils import (
    CONTACTS_TSV_PATH,
    CONTACTS_YAML_PATH,
    EMAIL_GITHUB_MAP,
    EMAIL_ORCID_MAP,
    EMAIL_WIKIDATA_MAP,
    ONE_YEAR_AGO,
    SKIP_EMAILS,
    get_github,
    get_ontologies,
    query_wikidata,
)


@lru_cache(maxsize=None)
def get_wikidata_from_github(
    github_id: str,
) -> tuple[None, None] | tuple[str, None] | tuple[str, str]:
    """Lookup bibliometric data from Wikidata using a GitHub handle."""
    query = dedent(
        f"""\
        SELECT ?item ?orcid 
        WHERE 
        {{
            ?item wdt:P2037 "{github_id}" .
            OPTIONAL {{ ?item wdt:P496 ?orcid }} .
        }}
    """
    )
    records = query_wikidata(query)
    if not records:
        return None, None
    record = records[0]
    wikidata_id = record["item"]["value"].removeprefix(
        "http://www.wikidata.org/entity/"
    )
    orcid_id = record.get("orcid", {}).get("value")
    if orcid_id is None:
        tqdm.write(f"No ORCID for https://bioregistry.io/wikidata:{wikidata_id}")
    return wikidata_id, orcid_id


@lru_cache(maxsize=None)
def get_last_event(user: str) -> Optional[datetime.datetime]:
    """Get the date and time of the most recent action for this user."""
    events = get_github(f"https://api.github.com/users/{user}/events")
    if not events:
        return None
    try:
        return dateparser.parse(events[0]["created_at"]).replace(tzinfo=None)
    except KeyError:
        tqdm.write(f"[{user}] error: {events}")
        return None


@click.command()
@click.option("--path", help="Path to local metadata", type=Path)
def main(path: Optional[Path]):
    """Generate the contact table."""
    counter = Counter()
    data = {}
    ontologies = defaultdict(list)
    it = tqdm(sorted(get_ontologies(path=path).items()))
    for obo_id, record in it:
        it.set_postfix(ontology=obo_id)
        contact = record.get("contact", {})
        github_id = contact.get("github")
        email = contact.get("email")
        if github_id is None and email is not None:
            github_id = EMAIL_GITHUB_MAP.get(email)
        if github_id is not None:
            key = "github", github_id.casefold()
            wikidata_id, orcid_id = get_wikidata_from_github(github_id)
            last_active = get_last_event(github_id)
        elif email is None or email in SKIP_EMAILS:
            continue
        else:
            key = "email", email.casefold()
            orcid_id = EMAIL_ORCID_MAP.get(email)
            wikidata_id = EMAIL_WIKIDATA_MAP.get(email)
            last_active = None

        counter[key] += 1
        ontologies[key].append(obo_id)
        data[key] = {
            "github": github_id,
            "label": contact["label"],
            "email": email,
            "wikidata": wikidata_id,
            "orcid": orcid_id,
            "last_active": last_active,
            "last_active_recent": ONE_YEAR_AGO < last_active if last_active else False,
        }

    # Output TSV
    df_rows = [dict(count=count, **data[key]) for key, count in counter.most_common()]
    df = pd.DataFrame(df_rows).sort_values(["count", "github"], ascending=[False, True])
    df.to_csv(CONTACTS_TSV_PATH, sep="\t", index=False)

    # Output YAML
    yaml_rows = [
        dict(
            ontologies=ontologies[
                ("github", row["github"].casefold())
                if row["github"]
                else ("email", row["email"].casefold())
            ],
            **row,
        )
        for row in df_rows
    ]
    with CONTACTS_YAML_PATH.open("w") as file:
        yaml.safe_dump(
            yaml_rows,
            file,
            allow_unicode=True,
        )


if __name__ == "__main__":
    main()
