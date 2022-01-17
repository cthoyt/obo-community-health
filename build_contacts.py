"""Build the contacts data table."""

from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Optional

import click
import dateparser
import pandas as pd
import yaml
from tqdm import tqdm

from utils import (CONTACTS_TSV_PATH, CONTACTS_YAML_PATH, ONE_YEAR_AGO,
                   get_github, get_ontologies, query_wikidata)


@lru_cache(maxsize=None)
def get_wikidata_from_github(github_id: str) -> tuple[str, Optional[str]]:
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
        print(f"No ORCID for https://bioregistry.io/wikidata:{wikidata_id}")
    return wikidata_id, orcid_id


@lru_cache(maxsize=None)
def get_last_event(user: str) -> Optional[datetime]:
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
        if record.get("is_obsolete") or record.get("activity_status") != "active":
            continue
        it.set_postfix(ontology=obo_id)
        contact = record["contact"]
        github_id = contact.get("github")
        if github_id is None:
            continue
        counter[github_id.casefold()] += 1
        ontologies[github_id.casefold()].append(obo_id)
        wikidata_id, orcid_id = get_wikidata_from_github(github_id)
        last_active = get_last_event(github_id)
        data[github_id.casefold()] = {
            "github": github_id,
            "label": contact["label"],
            "email": contact["email"],
            "wikidata": wikidata_id,
            "orcid": orcid_id,
            "last_active": last_active,
            "last_active_recent": ONE_YEAR_AGO < last_active if last_active else False,
        }

    # Output TSV
    df_rows = [
        dict(
            count=count,
            **data[github_id.casefold()],
        )
        for github_id, count in counter.most_common()
    ]
    df = pd.DataFrame(df_rows).sort_values(["count", "github"], ascending=[False, True])
    df.to_csv(CONTACTS_TSV_PATH, sep="\t", index=False)

    # Output YAML
    yaml_rows = [
        dict(
            ontologies=ontologies[row["github"].casefold()],
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
