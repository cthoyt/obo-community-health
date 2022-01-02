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

# Contacts data
CONTACTS_TSV_PATH = DATA.joinpath("contacts_table.tsv")
CONTACTS_YAML_PATH = DATA.joinpath("contacts.yaml")

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