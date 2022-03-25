import datetime
import json
import math
import pickle
from dataclasses import dataclass
from operator import attrgetter, itemgetter
from pathlib import Path
from typing import Iterable, Optional, TypeVar, Union

import bioregistry
import click
import dateparser
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from bioregistry import get_bioportal_prefix, get_ols_prefix, get_registry_invmap
from dataclasses_json import dataclass_json
from jinja2 import Environment, FileSystemLoader
from more_click import force_option, verbose_option
from tqdm import tqdm

from utils import (
    CONTACTS_YAML_PATH,
    DATA,
    EMAIL_GITHUB_MAP,
    EMAIL_ORCID_MAP,
    EMAIL_WIKIDATA_MAP,
    ODK_REPOS_PATH,
    ONE_YEAR_AGO,
    get_github,
    get_ontologies,
)

HERE = Path(__file__).parent.resolve()
TEMPLATES = HERE.joinpath("templates")

DOCS = HERE.joinpath("docs")
DOCS.mkdir(exist_ok=True, parents=True)
INDEX = DOCS.joinpath("index.html")
CONTACTS_PATH = DOCS.joinpath("contacts.html")
CONTACTS_CURATION_PATH = DOCS.joinpath("contacts_curation.html")

PATH_PICKLE = DATA.joinpath("data.pkl")
PATH_TSV = DATA.joinpath("data.tsv")
PATH_JSON = DATA.joinpath("data.json")
CONTACT_TRIVIA_PATH = DATA.joinpath("contacts_trivia.yaml")
PATH_HIST = DOCS.joinpath("score_histogram.png")
RESPONSIBILITY_HIST = DOCS.joinpath("responsibility_histogram.png")
ISSUE_SCATTER = DOCS.joinpath("score_issue_scatter.png")

environment = Environment(
    autoescape=True, loader=FileSystemLoader(TEMPLATES), trim_blocks=False
)
index_template = environment.get_template("index.html")
contacts_template = environment.get_template("contacts.html")
# ontology_template = environment.get_template("ontology.html")

PREFIX = "https://github.com/"
TITLE = "Add obofoundry topic to repo metadata"

GITHUB_BONUS = 3
Issue = TypeVar("Issue")
SOFTWARE_LICENSES = {"mit", "bsd-3-clause", "apache-2.0"}  # TODO


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


def iterate_repos(
    path: Optional[Path] = None,
) -> Iterable[
    tuple[str, str, Union[tuple[str, str], tuple[None, None]], dict[str, any]]
]:
    ontologies = get_ontologies(path=path)

    for obo_id, record in tqdm(sorted(ontologies.items()), desc="Processing OBO conf"):
        if record.get("is_obsolete") or record.get("activity_status") != "active":
            continue

        repository = record.get("repository")
        tracker = record.get("tracker")

        if repository is not None:
            if repository.startswith(PREFIX):
                owner, repo, *_ = repository[len(PREFIX) :].split("/")
                yield obo_id, record["title"], (owner, repo), record
            else:
                tqdm.write(f'no github repository for {record["id"]}: {tracker}')
                yield obo_id, record["title"], (None, None), record
        else:
            # All active ontologies have trackers. Most of them
            # have GitHub, but not all. Don't consider the non-GitHub
            # ones
            if not tracker:
                tqdm.write(f'no tracker for {record["id"]}')
                yield obo_id, record["title"], (None, None), record
            elif not tracker.startswith(PREFIX):
                tqdm.write(f'no github tracker for {record["id"]}: {tracker}')
                yield obo_id, record["title"], (None, None), record
            else:
                # Since we assume it's a GitHub link, slice out the prefix then
                # parse the owner and repository out of the path
                owner, repo, *_ = tracker[len(PREFIX) :].split("/")
                yield obo_id, record["title"], (owner, repo), record


@dataclass_json
@dataclass
class Result:
    prefix: str
    title: str
    description: str
    homepage: Optional[str]
    contact_label: str
    contact_email: str
    contact_github: Optional[str]
    contact_wikidata: Optional[str]
    contact_orcid: Optional[str]
    contact_recent: bool
    bioregistry_prefix: str
    bioportal_prefix: Optional[str]
    ols_prefix: Optional[str]

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

        score = adjust(
            score,
            self.contact_github is not None,
            errors,
            "missing contact's GitHub handle",
            # without this annotation, it's not possible to get in touch on GitHub programmatically
            punishment=5,
        )
        if self.contact_github is not None:
            score = adjust(
                score,
                self.contact_recent,
                errors,
                "contact is inactive",
                punishment=3,
            )
        score = adjust(
            score,
            self.contact_wikidata is not None,
            errors,
            "could not look up contact on WikiData via GitHub",
        )
        score = adjust(
            score,
            self.contact_orcid is not None,
            errors,
            "could not look up contact ORCID via WikiData",
        )
        score = adjust(
            score,
            self.homepage is not None,
            errors,
            "missing homepage",
            punishment=5,  # seriously?
        )
        score = adjust(
            score,
            self.bioregistry_prefix is not None,
            errors,
            "missing Bioregistry mapping",
            # without this annotation, the bioregistry isn't doing its job
            punishment=5,
        )
        score = adjust(
            score,
            self.bioportal_prefix is not None,
            errors,
            "missing BioPortal mapping",
            punishment=1,
        )
        score = adjust(
            score,
            self.ols_prefix is not None,
            errors,
            "missing OLS mapping",
            punishment=1,
        )
        return score, errors


@dataclass_json
@dataclass
class GithubResult(Result):
    owner: str
    repo: str
    repo_description: str
    stars: int
    license: str
    open_issues: int
    repo_homepage: str
    pushed_at: datetime.datetime
    pushed_last_year: bool
    has_obofoundry_topic: bool
    most_recent_datetime: datetime.datetime
    most_recent_number: str
    most_recent_last_year: bool
    odk_version: Optional[str]
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
            msg="missing obofoundry GitHub topic",
            punishment=5,  # severe - means no engagement with community
        )
        score = adjust(
            score,
            self.odk_version is not None,
            errors=errors,
            msg="not using ODK",
            punishment=3,
        )
        score = adjust(
            score, self.pushed_last_year, errors=errors, msg="not recently pushed"
        )

        # License
        if self.license is None:
            score -= 2
            errors.append("no LICENSE on GitHub")
        elif self.license == "other":
            score -= 2
            errors.append("non-standard LICENSE given on GitHub")
        elif self.license in SOFTWARE_LICENSES:
            score -= 3
            errors.append("inappropriate software LICENSE given on GitHub")
        else:  # Reward using well-recognized licenses.
            score += 2

        stars = self.stars
        if not stars:
            score += fslog10(stars, 3)

        contributions = self.lifetime_total_contributions
        if contributions:
            score += fslog10(self.lifetime_total_contributions, 2)

        contributors = self.lifetime_unique_contributors
        if contributors:
            score += fslog10(self.lifetime_unique_contributors, 4)

        return score, errors


def get_data(
    *,
    contacts,
    odk_repos,
    force: bool = False,
    test: bool = False,
    path: Optional[Path] = None,
) -> list[Result]:
    if PATH_PICKLE.is_file() and not force and not test:
        with PATH_PICKLE.open("rb") as file:
            return pickle.load(file)

    repos = sorted(iterate_repos(path=path))
    if test:
        c = 0
        repos = [
            r for r in repos if r[3] is None or (c := c + int(r[3] is not None)) <= 3
        ]
    rows: list[Result] = []
    repos = tqdm(repos, desc="Repositories")
    for prefix, title, (owner, repo), record in repos:
        repos.set_postfix(repo=f"{owner}/{repo}")
        description = record["description"]
        homepage = record.get("homepage")
        contact = record["contact"]
        contact_label = contact["label"]
        contact_email = contact["email"]
        contact_github = contact.get("github") or EMAIL_GITHUB_MAP.get(contact_email)
        contact_wikidata = contacts.get(contact_github, {}).get(
            "wikidata"
        ) or EMAIL_WIKIDATA_MAP.get(contact_email)
        contact_orcid = contacts.get(contact_github, {}).get(
            "orcid"
        ) or EMAIL_ORCID_MAP.get(contact_email)
        contact_recent = contacts.get(contact_github, {}).get(
            "last_active_recent", False
        )

        # External
        pp = record["preferredPrefix"]
        bioregistry_prefix = get_registry_invmap("obofoundry").get(pp)
        if bioregistry_prefix is None:
            tqdm.write(f"No bioregistry prefix for {pp}")
            bioportal_prefix = None
            ols_prefix = None
        else:
            bioportal_prefix = get_bioportal_prefix(bioregistry_prefix)
            ols_prefix = get_ols_prefix(bioregistry_prefix)

        if owner is None:
            rows.append(
                Result(
                    prefix=prefix,
                    title=title,
                    description=description,
                    homepage=homepage,
                    contact_github=contact_github,
                    contact_email=contact_email,
                    contact_label=contact_label,
                    contact_wikidata=contact_wikidata,
                    contact_orcid=contact_orcid,
                    contact_recent=contact_recent,
                    bioregistry_prefix=bioregistry_prefix,
                    bioportal_prefix=bioportal_prefix,
                    ols_prefix=ols_prefix,
                )
            )
            continue
        info = get_info(owner, repo)
        repo_description = info["description"]
        stars = info["stargazers_count"]
        license = info["license"]
        open_issues = info["open_issues"]
        repo_homepage = info["homepage"]
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
                description=description,
                homepage=homepage,
                contact_github=contact_github,
                contact_email=contact_email,
                contact_label=contact_label,
                contact_wikidata=contact_wikidata,
                contact_orcid=contact_orcid,
                contact_recent=contact_recent,
                bioregistry_prefix=bioregistry_prefix,
                bioportal_prefix=bioportal_prefix,
                ols_prefix=ols_prefix,
                owner=owner,
                repo=repo,
                repo_description=repo_description,
                stars=stars,
                license=license["key"] if license else None,
                open_issues=open_issues,
                repo_homepage=repo_homepage,
                pushed_at=pushed_at,
                pushed_last_year=pushed_last_year,
                has_obofoundry_topic=has_obofoundry_topic,
                odk_version=odk_repos.get(f"{owner}/{repo}"),
                most_recent_datetime=most_recent_datetime,
                most_recent_number=most_recent_updated_number,
                most_recent_last_year=update_last_year,
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

    rows = sorted(rows, key=attrgetter("prefix"))

    pd.DataFrame(rows).to_csv(PATH_TSV, sep="\t", index=False)
    with PATH_PICKLE.open("wb") as file:
        pickle.dump(rows, file)
    return rows


@click.command()
@force_option
@verbose_option
@click.option("--test", is_flag=True)
@click.option("--path", help="Path to local metadata", type=Path)
def main(force: bool, test: bool, path):
    with CONTACTS_YAML_PATH.open() as file:
        contacts = {record["github"]: record for record in yaml.safe_load(file)}

    odk_repos_df = pd.read_csv(ODK_REPOS_PATH, sep="\t")
    odk_repos = dict(odk_repos_df[["repository", "version"]].values)

    rows = get_data(
        contacts=contacts, odk_repos=odk_repos, force=force, test=test, path=path
    )
    with PATH_JSON.open("w") as file:
        json.dump(
            {
                row.prefix: {
                    **dict(zip(("score", "messages"), row.get_score())),
                    **row.to_dict(),
                }
                for row in rows
            },
            file,
            indent=2,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        )

    # Author responsibility histogram
    counts = [contact["count"] for contact in contacts.values()]
    responsible_one, responsible_multiple, responsible_multiple_sum = 0, 0, 0
    for count in counts:
        if count == 1:
            responsible_one += 1
        else:
            responsible_multiple += 1
            responsible_multiple_sum += count

    print(f"People: {len(counts)}")
    has_github = sum(contact.get("github") is not None for contact in contacts.values())
    print(
        f"  w/ GitHub: {has_github}/{len(contacts)} ({has_github / len(contacts):.2%})"
    )
    has_wikidata = sum(
        contact.get("wikidata") is not None for contact in contacts.values()
    )
    print(
        f"  w/ Wikidata: {has_wikidata}/{len(contacts)} ({has_wikidata / len(contacts):.2%})"
    )
    has_orcid = sum(contact.get("orcid") is not None for contact in contacts.values())
    print(f"  w/ ORCID: {has_orcid}/{len(contacts)} ({has_orcid / len(contacts):.2%})")
    print(
        f"  responsible for one ontology:"
        f" {responsible_one}/{len(counts)} ({responsible_one / len(counts):.2%})"
    )
    print(
        f"  responsible for two or more ontologies:"
        f" {responsible_multiple}/{len(counts)} ({responsible_multiple / len(counts):.2%})"
    )
    active_contacts = sum(
        contact["last_active_recent"] for contact in contacts.values()
    )
    print(
        f"  active on GitHub (last year):"
        f" {active_contacts}/{has_github} ({active_contacts / has_github:.2%})"
    )
    inactive_contacts = has_github - active_contacts
    print(
        f"  inactive on GitHub (last year):"
        f" {inactive_contacts}/{has_github} ({inactive_contacts / has_github:.2%})"
    )

    print(
        f"Ontologies (non-inactive, non-obsolete, non-orphaned, w/ GitHub): {len(rows)}"
    )
    print(
        f"  w/ responsible person who's responsible for one ontology:"
        f" {responsible_one}/{sum(counts)} ({responsible_one / sum(counts):.2%})"
    )
    print(
        f"  w/ responsible person who's responsible for two or more ontologies:"
        f" {responsible_multiple_sum}/{sum(counts)} ({responsible_multiple_sum / sum(counts):.2%})"
    )

    active_ontologies = sum(
        len(contact["ontologies"])
        for contact in contacts.values()
        if contact["last_active_recent"]
    )
    inactive_ontologies = sum(counts) - active_ontologies

    print(
        f"  w/ responsible person who's inactive on GitHub (last year):"
        f" {inactive_ontologies}/{len(rows)} ({inactive_contacts / len(rows):.2%})"
    )

    CONTACT_TRIVIA_PATH.write_text(
        yaml.dump(
            {
                "number_responsibles": len(counts),
                "single_responsibles": responsible_one,
                "multiple_responsibles": responsible_multiple,
                "number_ontologies": sum(counts),
                "single_ontologies": responsible_one,
                "multiple_ontologies": responsible_multiple_sum,
                "active_responsibles": active_contacts,
                "inactive_responeibles": len(contacts) - active_contacts,
                "active_ontologies": active_ontologies,
                "inactive_ontologies": inactive_ontologies,
            }
        )
    )

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(counts, ax=ax)
    ax.set_xlabel("Number Responsible Ontologies")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(RESPONSIBILITY_HIST, dpi=300)

    # Score histogram
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
    sns.scatterplot(x=x, y=y, ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Open Issues")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(ISSUE_SCATTER, dpi=300)

    today = datetime.date.today()

    INDEX.write_text(index_template.render(rows=rows, today=today))

    CONTACTS_PATH.write_text(
        contacts_template.render(
            bioregistry=bioregistry,
            curation=False,
            rows=list(contacts.values()),
            today=today,
        )
    )
    CONTACTS_CURATION_PATH.write_text(
        contacts_template.render(
            bioregistry=bioregistry,
            curation=True,
            rows=[row for row in contacts.values() if not row.get("wikidata")],
            today=today,
        )
    )

    no_orcid_emails = ", ".join(
        sorted(row["email"] for row in contacts.values() if not row.get("wikidata"))
    )
    if no_orcid_emails:
        click.echo("These people don't have ORCID annotations:")
        click.echo(no_orcid_emails)

    # for row in rows:
    #     ontology_html = ontology_template.render(row=row)
    #     directory = DOCS.joinpath(row.prefix)
    #     directory.mkdir(exist_ok=True, parents=True)
    #     with directory.joinpath("index.html").open("w") as file:
    #         print(ontology_html, file=file)


if __name__ == "__main__":
    main()
