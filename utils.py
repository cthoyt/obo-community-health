"""Utilities for OBO Community Health Assessment."""

import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pystow
import requests
import yaml
from ratelimit import rate_limited

__all__ = [
    "get_github",
    "get_ontologies",
    "query_wikidata",
]

HERE = Path(__file__).parent.resolve()

DATA = HERE.joinpath("data")
DATA.mkdir(exist_ok=True, parents=True)

# Repo data
REPO_DATA_PICKLE = DATA.joinpath("data.pkl")
REPO_DATA_TSV = DATA.joinpath("data.tsv")
REPO_DATA_JSON = DATA.joinpath("data.json")

# Contacts data
CONTACTS_TSV_PATH = DATA.joinpath("contacts_table.tsv")
CONTACTS_YAML_PATH = DATA.joinpath("contacts.yaml")

ODK_REPOS_PATH = DATA.joinpath("odk_repos.tsv")

# Load the GitHub access token via PyStow. We'll
# need it so we don't hit the rate limit
TOKEN = pystow.get_config("github", "token", raise_on_missing=True)

#: URL for downloading OBO Foundry metatada
URL = "https://raw.githubusercontent.com/OBOFoundry/OBOFoundry.github.io/master/_config.yml"

#: WikiData SPARQL endpoint. See https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service#Interfacing
WIKIDATA_SPARQL = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
WIKIDATA_HEADERS = {
    "User-Agent": f"obo-community-health/1.0",
}

NOW = datetime.datetime.now()
ONE_YEAR_AGO = NOW - datetime.timedelta(weeks=52)
FIVE_YEARS_AGO = NOW - datetime.timedelta(weeks=52 * 5)


EMAIL_GITHUB_MAP = {
    "peteremidford@yahoo.com": "pmidford",
    "cjmungall@lbl.gov": "cmungall",
    "wasila.dahdul@usd.edu": "wdahdul",
    "mcourtot@gmail.com": "mcourtot",
    "a.chang@tu-bs.de": "BRENDA-Enzymes",
    "engelsta@ohsu.edu": "markengelstad",
    "Lindsay.Cowell@utsouthwestern.edu": "lgcowell",
    "n.lenovere@gmail.com": "gambardella",
    "mbrochhausen@gmail.com": "mbrochhausen",
    "cherry@genome.stanford.edu": "jmcherry-zz",
    "BatchelorC@rsc.org": "batchelorc",
    "stoeckrt@pcbi.upenn.edu": "cstoeckert",
}
EMAIL_ORCID_MAP = {
    "Leszek@missouri.edu": "0000-0002-9316-2919",
    "nicolas@ascistance.co.uk": "0000-0002-6309-7327",
}
EMAIL_WIKIDATA_MAP = {
    "Leszek@missouri.edu": "Q110623916",
    "nicolas@ascistance.co.uk": "Q21055156",
}
#: These emails are skipped, mostly because they are group emails.
SKIP_EMAILS = {
    "evoc@sanbi.ac.za",
    "adw_geeks@umich.edu",
    "po-discuss@plantontology.org",
    "curator@inoh.org",
    "interhelp@ebi.ac.uk",
    "psidev-gps-dev@lists.sourceforge.net",
}


@rate_limited(calls=5_000, period=60 * 60)
def get_github(
    url: str, accept: Optional[str] = None, params: Optional[dict[str, any]] = None
):
    headers = {
        "Authorization": f"token {TOKEN}",
    }
    if accept:
        headers["Accept"] = accept
    return requests.get(url, headers=headers, params=params).json()


def get_ontologies(path: Optional[Path] = None) -> dict[str, dict[str, any]]:
    if path is None:
        return get_cached_ontologies()
    with path.open() as file:
        return _get_ontology_helper(file)


@lru_cache
def get_cached_ontologies() -> dict[str, dict[str, any]]:
    # List of ontologies and associated metadata from OBO Foundry
    res = requests.get(URL)
    return _get_ontology_helper(res.content)


def _get_ontology_helper(content):
    parsed_res = yaml.safe_load(content)
    return {entry["id"]: entry for entry in parsed_res["ontologies"]}


def query_wikidata(query: str):
    """Query the Wikidata SPARQL endpoint and return JSON."""
    res = requests.get(
        WIKIDATA_SPARQL,
        params={"query": query, "format": "json"},
        headers=WIKIDATA_HEADERS,
    )
    res.raise_for_status()
    res_json = res.json()
    return res_json["results"]["bindings"]
