# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "cachier",
#     "click",
#     "dateparser",
#     "pandas",
#     "pystow",
#     "pyyaml",
#     "ratelimit",
#     "requests",
#     "tqdm",
#     "wikidata-client",
# ]
# ///

"""Build the contacts data table."""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from pathlib import Path

import click
import dateparser
import pandas as pd
import wikidata_client
import yaml
from cachier import cachier
from pystow.github import get_user_events
from tqdm import tqdm

from utils import (
    CONTACTS_TSV_PATH,
    CONTACTS_YAML_PATH,
    EMAIL_GITHUB_MAP,
    EMAIL_ORCID_MAP,
    EMAIL_WIKIDATA_MAP,
    GITHUB_REMAP,
    ONE_YEAR_AGO,
    SKIP_EMAILS,
    get_ontologies,
)


@cachier(stale_after=datetime.timedelta(days=1), allow_none=False)
def get_last_event(user: str) -> datetime.datetime | None:
    """Get the date and time of the most recent action for this user."""
    events = get_user_events(user).json()
    if not events:
        return None
    try:
        return dateparser.parse(events[0]["created_at"]).replace(tzinfo=None)
    except KeyError:
        tqdm.write(f"[{user}] error: {events}")
        return None


get_entity_by_github = cachier(stale_after=datetime.timedelta(days=30))(
    wikidata_client.get_entity_by_github
)


@click.command()
@click.option("--path", help="Path to local metadata", type=Path)
def main(path: Path | None):
    """Generate the contact table."""
    counter = Counter()
    data = {}
    ontologies = defaultdict(list)
    it = tqdm(sorted(get_ontologies(path=path).items()))
    for obo_id, record in it:
        it.set_postfix(ontology=obo_id)
        contact = record.get("contact", {})
        if not contact:
            # only the case for deprecated ontologies
            continue

        email = contact.get("email")
        if email in SKIP_EMAILS:
            continue
        if email is None:
            # this is only the case for REX
            continue

        github_id = contact.get("github") or EMAIL_GITHUB_MAP.get(email)
        if github_id:
            github_id = GITHUB_REMAP.get(github_id, github_id)

        orcid_id = contact.get("orcid") or EMAIL_ORCID_MAP.get(email)

        if github_id is not None:
            key = "github", github_id.casefold()
            wikidata_id = EMAIL_WIKIDATA_MAP.get(email) or get_entity_by_github(github_id)
            last_active = get_last_event(github_id)
        else:
            key = "email", email.casefold()
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
