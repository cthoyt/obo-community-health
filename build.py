import datetime
import math
import pickle
from dataclasses import dataclass
from functools import lru_cache
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Optional, TypeVar, Union

import click
import dateparser
import matplotlib.pyplot as plt
import pandas as pd
import pystow
import requests
import seaborn as sns
import yaml
from dataclasses_json import dataclass_json
from jinja2 import Environment, FileSystemLoader
from more_click import force_option, verbose_option
from ratelimit import rate_limited
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()
TEMPLATES = HERE.joinpath("templates")

DOCS = HERE.joinpath("docs")
DOCS.mkdir(exist_ok=True, parents=True)
INDEX = DOCS.joinpath("index.html")

DATA = HERE.joinpath("data")
DATA.mkdir(exist_ok=True, parents=True)
PATH_PICKLE = DATA.joinpath("data.pkl")
PATH_TSV = DATA.joinpath("data.tsv")
PATH_JSON = DATA.joinpath("data.json")
PATH_HIST = DOCS.joinpath("score_histogram.png")
ISSUE_SCATTER = DOCS.joinpath("score_issue_scatter.png")

environment = Environment(
    autoescape=True, loader=FileSystemLoader(TEMPLATES), trim_blocks=False
)
index_template = environment.get_template("index.html")
# ontology_template = environment.get_template("ontology.html")

URL = "https://raw.githubusercontent.com/OBOFoundry/OBOFoundry.github.io/master/_config.yml"
PREFIX = "https://github.com/"
TITLE = "Add obofoundry topic to repo metadata"

# Load the GitHub access token via PyStow. We'll
# need it so we don't hit the rate limit
TOKEN = pystow.get_config("github", "token", raise_on_missing=True)

NOW = datetime.datetime.now()
ONE_YEAR_AGO = NOW - datetime.timedelta(weeks=52)
GITHUB_BONUS = 3
Issue = TypeVar("Issue")
SOFTWARE_LICENSES = {"mit", "bsd-3-clause", "apache-2.0"}  # TODO


@lru_cache
def get_ontologies() -> dict[str, dict[str, any]]:
    # List of ontologies and associated metadata from OBO Foundry
    res = requests.get(URL)
    return _get_ontology_helper(res.content)


def _get_ontology_helper(content):
    parsed_res = yaml.safe_load(content)
    return {entry["id"]: entry for entry in parsed_res["ontologies"]}


def floor(x: float) -> int:
    return int(math.floor(x))


def slog10(x: float) -> float:
    """Apply the log10 but in a safe way for zeros."""
    return math.log10(x + 1) - 1


def fslog10(x: float, m: float = 1.0) -> int:
    return floor(m * slog10(x))


def adjust(
    score: int,
    decision: bool,
    errors: list[str],
    msg: str,
    *,
    reward: int = 1,
    punishment: int = 0,
) -> int:
    """Adjust the score based on a given decision."""
    if decision:
        score += reward
    else:
        score -= punishment
        errors.append(msg)
    return score


def get_most_recent_updated_issue(owner: str, repo: str) -> Optional[Issue]:
    return get_first_issue(
        owner=owner,
        repo=repo,
        params={"state": "all", "sort": "updated", "direction": "desc"},
    )


def get_oldest_open_issue(owner: str, repo: str) -> Optional[Issue]:
    return get_first_issue(
        owner=owner,
        repo=repo,
        params={"state": "open", "sort": "updated", "direction": "asc"},
    )


def get_first_issue(owner: str, repo: str, params) -> Optional[Issue]:
    issues = get_issues(owner=owner, repo=repo, params=params)
    if issues:
        return issues[0]


def get_contributions(owner: str, repo: str, params=None):
    return get_github(
        f"https://api.github.com/repos/{owner}/{repo}/stats/contributors", params=params
    )


def get_last_year_contributions(owner: str, repo: str, params=None):
    return get_github(
        f"https://api.github.com/repos/{owner}/{repo}/stats/commit_activity",
        params=params,
    )


def get_issues(owner: str, repo: str, params=None):
    return get_github(
        f"https://api.github.com/repos/{owner}/{repo}/issues", params=params
    )


def get_info(owner: str, repo: str):
    return get_github(f"https://api.github.com/repos/{owner}/{repo}")


def get_topics(owner: str, repo: str):
    # Get all topics from the repository. See more information on the GH docs:
    # https://docs.github.com/en/rest/reference/repos#get-all-repository-topics
    rv = get_github(
        f"https://api.github.com/repos/{owner}/{repo}/topics",
        accept="application/vnd.github.mercy-preview+json",
    )
    return rv["names"]


def has_odk_config(owner: str, repo: str) -> bool:
    rv = get_github(
        f"https://api.github.com/search/code?q=repo:{owner}/{repo}+filename:odk.yaml",
    )
    try:
        return 0 < rv["total_count"]
    except KeyError:
        print(f"failed to search ODK config for {owner}/{repo}:", rv)
        return False


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


def iterate_repos(
    ontologies,
) -> Iterable[Union[tuple[str, str, str, str], tuple[str, str, None, None]]]:
    for obo_id, record in tqdm(sorted(ontologies.items()), desc="Processing OBO conf"):
        if record.get("is_obsolete") or record.get("activity_status") != "active":
            continue

        repository = record.get("repository")
        tracker = record.get("tracker")

        if repository is not None:
            if repository.startswith(PREFIX):
                owner, repo, *_ = repository[len(PREFIX) :].split("/")
                yield obo_id, record["title"], owner, repo
            else:
                tqdm.write(f'no github repository for {record["id"]}: {tracker}')
                yield obo_id, record["title"], None, None
        else:
            # All active ontologies have trackers. Most of them
            # have GitHub, but not all. Don't consider the non-GitHub
            # ones
            if not tracker:
                tqdm.write(f'no tracker for {record["id"]}')
                yield obo_id, record["title"], None, None
            elif not tracker.startswith(PREFIX):
                tqdm.write(f'no github tracker for {record["id"]}: {tracker}')
                yield obo_id, record["title"], None, None
            else:
                # Since we assume it's a GitHub link, slice out the prefix then
                # parse the owner and repository out of the path
                owner, repo, *_ = tracker[len(PREFIX) :].split("/")
                yield obo_id, record["title"], owner, repo


@dataclass_json
@dataclass
class Result:
    prefix: str
    title: str
    contact_label: str
    contact_email: str
    contact_github: Optional[str]

    def get_score(self) -> tuple[int, list[str]]:
        score = 0
        errors = []
        # Bad naming
        if f"({self.prefix.lower()})" in self.title.lower():
            score -= 5
            errors.append("title contains prefix")
        elif f"{self.prefix.lower()}:" in self.title.lower():
            score -= 5
            errors.append("title contains prefix")
        elif f"{self.prefix.lower()} -" in self.title.lower():
            score -= 5
            errors.append("title contains prefix")
            score -= 5
        elif f"{self.prefix.lower} ontology" == self.title.lower():
            score -= 5
            errors.append("title is redundant of prefix")
        elif self.prefix.casefold() == self.title.casefold():
            score -= 5
            errors.append("title is redundant of prefix")
        else:
            score += 5

        score += adjust(
            score,
            self.contact_github is not None,
            errors,
            "missing github contact",
            # without this annotation, it's not possible to get in touch on GitHub programmatically
            punishment=5,
        )
        return score, errors


@dataclass_json
@dataclass
class GithubResult(Result):
    owner: str
    repo: str
    description: str
    stars: int
    license: str
    open_issues: int
    homepage: str
    pushed_at: datetime.datetime
    pushed_last_year: bool
    has_obofoundry_topic: bool
    most_recent_datetime: datetime.datetime
    most_recent_number: str
    most_recent_last_year: bool
    # has_odk: bool
    # lifetime
    lifetime_total_contributions: int
    lifetime_unique_contributors: int
    top_lifetime_contributor: int
    top_lifetime_contributions: int
    # last year
    last_year_total_contributions: int
    last_year_unique_contributors: int
    top_last_year_contributor: int
    top_last_year_contributions: int

    def get_score(self) -> tuple[int, list[str]]:
        score, errors = super().get_score()

        score += 2 * GITHUB_BONUS
        score = adjust(
            score,
            self.has_obofoundry_topic,
            errors=errors,
            msg="missing obofoundry github topic",
            punishment=5,  # severe - means no engagement with community
        )
        score = adjust(
            score, self.pushed_last_year, errors=errors, msg="not recently pushed"
        )
        # rv = j(rv, self.has_odk)

        # License
        if self.license is None:
            score -= 2
            errors.append("no LICENSE on GitHub")
        elif self.license == "other":
            score -= 1
            errors.append("non-standard LICENSE given on GitHub")
        elif self.license in SOFTWARE_LICENSES:
            score -= 1
            errors.append("inappropriate software LICENSE given on GitHub")
        else:  # Reward using well-recognized licenses.
            score += 1

        stars = self.stars
        if not stars:
            score += fslog10(stars, 2)

        contributions = self.lifetime_total_contributions
        if contributions:
            score += fslog10(self.lifetime_total_contributions, 2)

        contributors = self.lifetime_unique_contributors
        if contributors:
            score += fslog10(self.lifetime_unique_contributors, 4)

        return score, errors


def get_data(
    force: bool = False, test: bool = False, path: Optional[Path] = None
) -> list[Result]:
    if PATH_PICKLE.is_file() and not force and not test:
        with PATH_PICKLE.open("rb") as file:
            return pickle.load(file)

    if path is None:
        ontologies = get_ontologies()
    else:
        with path.open() as file:
            ontologies = _get_ontology_helper(file)

    repos = sorted(iterate_repos(ontologies))
    if test:
        c = 0
        repos = [
            r for r in repos if r[3] is None or (c := c + int(r[3] is not None)) <= 3
        ]
    rows: list[Result] = []
    repos = tqdm(repos, desc="Repositories")
    for prefix, title, owner, repo in repos:
        repos.set_postfix(repo=f"{owner}/{repo}")
        contact = ontologies[prefix].get("contact", {})
        contact_github = contact.get("github")
        contact_label = contact.get("label")
        contact_email = contact.get("email")

        if owner is None:
            rows.append(
                Result(
                    prefix=prefix,
                    title=title,
                    contact_github=contact_github,
                    contact_email=contact_email,
                    contact_label=contact_label,
                )
            )
            continue
        info = get_info(owner, repo)
        description = info["description"]
        stars = info["stargazers_count"]
        license = info["license"]
        open_issues = info["open_issues"]
        homepage = info["homepage"]
        pushed_at = dateparser.parse(info["pushed_at"]).replace(tzinfo=None)
        pushed_last_year = ONE_YEAR_AGO < pushed_at
        topics = get_topics(owner, repo)
        has_obofoundry_topic = "obofoundry" in topics
        if (
            most_recent_updated := get_most_recent_updated_issue(owner, repo)
        ) is not None:
            most_recent_datetime = dateparser.parse(
                most_recent_updated["updated_at"]
            ).replace(tzinfo=None)
            most_recent_updated_number = most_recent_updated["number"]
            update_last_year = ONE_YEAR_AGO < most_recent_datetime
        else:
            most_recent_datetime = None
            most_recent_updated_number = None
            update_last_year = False

        lifetime_contributions_ = get_contributions(owner, repo)
        lifetime_contributions = {
            entry["author"]["login"]: entry["total"]
            for entry in lifetime_contributions_
        }

        if lifetime_contributions:
            top_lifetime_contributor, top_lifetime_contributions = max(
                lifetime_contributions.items(), key=itemgetter(1)
            )
        else:
            top_lifetime_contributor, top_lifetime_contributions = None, None
        lifetime_total_contributions = sum(lifetime_contributions.values())
        lifetime_unique_contributors = len(lifetime_contributions)

        last_year_contributions = {
            entry["author"]["login"]: sum(
                week["c"]
                for week in entry["weeks"]
                if ONE_YEAR_AGO < datetime.datetime.utcfromtimestamp(week["w"])
            )
            for entry in lifetime_contributions_
        }

        if last_year_contributions:
            top_last_year_contributor, top_last_year_contributions = max(
                last_year_contributions.items(), key=itemgetter(1)
            )
        else:
            top_last_year_contributor, top_last_year_contributions = None, None
        last_year_unique_contributors = len(last_year_contributions)

        # last year contributions
        # https://docs.github.com/en/rest/reference/repos#get-the-last-year-of-commit-activity
        last_year_contributions = get_last_year_contributions(owner, repo)
        last_year_total_contributions = sum(
            entry["total"] for entry in last_year_contributions
        )

        # when was the last issue closed?
        rows.append(
            GithubResult(
                prefix=prefix,
                title=title,
                contact_github=contact_github,
                contact_email=contact_email,
                contact_label=contact_label,
                owner=owner,
                repo=repo,
                description=description,
                stars=stars,
                license=license["key"] if license else None,
                open_issues=open_issues,
                homepage=homepage,
                pushed_at=pushed_at,
                pushed_last_year=pushed_last_year,
                has_obofoundry_topic=has_obofoundry_topic,
                most_recent_datetime=most_recent_datetime,
                most_recent_number=most_recent_updated_number,
                most_recent_last_year=update_last_year,
                # has_odk=has_odk_config(owner, repo),
                # lifetime
                lifetime_total_contributions=lifetime_total_contributions,
                lifetime_unique_contributors=lifetime_unique_contributors,
                top_lifetime_contributor=top_lifetime_contributor,
                top_lifetime_contributions=top_lifetime_contributions,
                # last year
                last_year_total_contributions=last_year_total_contributions,
                last_year_unique_contributors=last_year_unique_contributors,
                top_last_year_contributor=top_last_year_contributor,
                top_last_year_contributions=top_last_year_contributions,
            )
        )

    rows = sorted(rows, key=lambda row: row.get_score()[0], reverse=True)

    pd.DataFrame(rows).to_csv(PATH_TSV, sep="\t", index=False)
    with PATH_PICKLE.open("wb") as file:
        pickle.dump(rows, file)
    # with PATH_JSON.open("w") as file:
    #     json.dump(rows, file)

    return rows


@click.command()
@force_option
@verbose_option
@click.option("--test", is_flag=True)
@click.option("--path", help="Path to local metadata", type=Path)
def main(force: bool, test: bool, path):
    rows = get_data(force=force, test=test, path=path)

    scores = [row.get_score()[0] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(scores, ax=ax)
    ax.set_xlabel("Score")
    fig.tight_layout()
    fig.savefig(PATH_HIST, dpi=300)

    # Correlation between score and number of issues
    fig, ax = plt.subplots(figsize=(8, 3))
    x, y = zip(
        *(
            (row.get_score()[0], row.open_issues)
            for row in rows
            if isinstance(row, GithubResult)
        )
    )
    sns.scatterplot(x, y, ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Open Issues")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(ISSUE_SCATTER, dpi=300)

    index_html = index_template.render(rows=rows)
    with INDEX.open("w") as file:
        print(index_html, file=file)

    # for row in rows:
    #     ontology_html = ontology_template.render(row=row)
    #     directory = DOCS.joinpath(row.prefix)
    #     directory.mkdir(exist_ok=True, parents=True)
    #     with directory.joinpath("index.html").open("w") as file:
    #         print(ontology_html, file=file)


if __name__ == "__main__":
    main()
