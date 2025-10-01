"""Utilities for OBO Community Health Assessment."""

import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pystow
import requests
import yaml

__all__ = [
    "get_ontologies",
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
ODK_REPOS_YAML_PATH = DATA.joinpath("odk_repos.yaml")

# Load the GitHub access token via PyStow. We'll
# need it so we don't hit the rate limit
TOKEN = pystow.get_config("github", "token", raise_on_missing=True)

#: URL for downloading OBO Foundry metatada
URL = "https://raw.githubusercontent.com/OBOFoundry/OBOFoundry.github.io/master/_config.yml"

#: WikiData SPARQL endpoint. See https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service#Interfacing
WIKIDATA_SPARQL = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
WIKIDATA_HEADERS = {
    "User-Agent": "obo-community-health/1.0",
}

NOW = datetime.datetime.now()
ONE_YEAR_AGO = NOW - datetime.timedelta(weeks=52)
FIVE_YEARS_AGO = NOW - datetime.timedelta(weeks=52 * 5)

# TODO
#  adeans@psu.edu, dal.alghamdi92@gmail.com, henrich@embl.de, jmcl@ebi.ac.uk, jmwhorton@uams.edu,
#  lucas.leclere@obs-banyuls.fr, mauno.vihinen@med.lu.se, mcmelek@msn.com, meghan.balk@gmail.com,
#  muamith@utmb.edu, noreply@example.com, xyz19940216@163.com
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
GITHUB_REMAP = {
    "megbalk": "meghalithic",
}
EMAIL_ORCID_MAP = {
    "julie@igbmc.u-strasbg.fr": "0000-0003-4893-3478",
    "Leszek@missouri.edu": "0000-0002-9316-2919",
    "nicolas@ascistance.co.uk": "0000-0002-6309-7327",
    "dal.alghamdi92@gmail.com": "0000-0002-2801-0767",
    "jmcl@ebi.ac.uk": "0000-0002-8361-2795",
    "maria.herrero@kcl.ac.uk": "0000-0001-7793-3296",
    "burkesquires@gmail.com": "0000-0001-9666-6285",
    "dsonensh@odu.edu": "0000-0001-9370-918X",
}
EMAIL_WIKIDATA_MAP = {
    "smtifahim@gmail.com": "Q57678362",
    "julie@igbmc.u-strasbg.fr": "Q91782245",
    "maria.herrero@kcl.ac.uk": "Q91270494",
    "Leszek@missouri.edu": "Q110623916",
    "nicolas@ascistance.co.uk": "Q21055156",
    "david.c.blackburn@gmail.com": "Q19978490",
    "lucas.leclere@obs-banyuls.fr": "Q83382941",
    "mcmelek@msn.com": "Q58853917",
    "muamith@utmb.edu": "Q61126869",
    "adeans@psu.edu": "Q21073927",
    "dal.alghamdi92@gmail.com": "Q136402381",
    "jmcl@ebi.ac.uk": "Q92202696",
    "john.garavelli@ebi.ac.uk": "Q56949596",
    "bakerc@unb.ca": "Q59389381",
    "meghan.balk@gmail.com": "Q59763318",
    "mauno.vihinen@med.lu.se": "Q38591073",
}
#: These emails are skipped, mostly because they are group emails.
SKIP_EMAILS = {
    "evoc@sanbi.ac.za",
    "adw_geeks@umich.edu",
    "po-discuss@plantontology.org",
    "curator@inoh.org",
    "interhelp@ebi.ac.uk",
    "psidev-gps-dev@lists.sourceforge.net",
    "noreply@example.com",
}


def get_ontologies(path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """Get the ontology dict."""
    if path is None:
        return get_cached_ontologies()
    with path.open() as file:
        return _get_ontology_helper(file)


@lru_cache
def get_cached_ontologies() -> dict[str, dict[str, Any]]:
    """Get the ontology dict."""
    # List of ontologies and associated metadata from OBO Foundry
    res = requests.get(URL, timeout=15)
    return _get_ontology_helper(res.content)


def _get_ontology_helper(content):
    parsed_res = yaml.safe_load(content)
    return {entry["id"]: entry for entry in parsed_res["ontologies"]}
