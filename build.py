import datetime
import json
import pathlib
from operator import itemgetter
from typing import Iterable, Union

import click
import dateparser
import pandas as pd
import pystow
import requests
import yaml
from jinja2 import Environment, FileSystemLoader
from more_click import force_option, verbose_option
from tqdm import tqdm

HERE = pathlib.Path(__file__).parent.resolve()
TEMPLATES = HERE.joinpath("templates")
DOCS = HERE.joinpath("docs")
DOCS.mkdir(exist_ok=True, parents=True)
PATH_TSV = HERE.joinpath("data.tsv")
PATH_JSON = HERE.joinpath("data.json")
INDEX = DOCS.joinpath("index.html")

environment = Environment(
    autoescape=True, loader=FileSystemLoader(TEMPLATES), trim_blocks=False
)
index_template = environment.get_template("index.html")
ontology_template = environment.get_template("ontology.html")

URL = "https://raw.githubusercontent.com/OBOFoundry/OBOFoundry.github.io/master/_config.yml"
PREFIX = "https://github.com/"
TITLE = "Add obofoundry topic to repo metadata"

# Load the GitHub access token via PyStow. We'll
# need it so we don't hit the rate limit
TOKEN = pystow.get_config("github", "token", raise_on_missing=True)

NOW = datetime.datetime.now()
ONE_YEAR_AGO = NOW - datetime.timedelta(weeks=52)


def get_most_recent_updated_issue(owner, repo):
    issues = get_issues(
        owner, repo, params={"state": "all", "sort": "updated", "direction": "desc"}
    )
    if issues:
        return issues[0]


def get_oldest_open_issue(owner, repo):
    issues = get_issues(
        owner, repo, params={"state": "open", "sort": "updated", "direction": "asc"}
    )
    if issues:
        return issues[0]


def get_contributions(owner, repo, params=None):
    return get_github(
        f"https://api.github.com/repos/{owner}/{repo}/stats/contributors", params=params
    )


def get_last_year_contributions(owner, repo, params=None):
    return get_github(
        f"https://api.github.com/repos/{owner}/{repo}/stats/commit_activity",
        params=params,
    )


def get_issues(owner, repo, params=None):
    return get_github(
        f"https://api.github.com/repos/{owner}/{repo}/issues", params=params
    )


def get_info(owner, repo):
    return get_github(f"https://api.github.com/repos/{owner}/{repo}")


def get_topics(owner, repo):
    # Get all topics from the repository. See more information on the GH docs:
    # https://docs.github.com/en/rest/reference/repos#get-all-repository-topics
    rv = get_github(
        f"https://api.github.com/repos/{owner}/{repo}/topics",
        accept="application/vnd.github.mercy-preview+json",
    )
    return rv["names"]


def has_odk_config(owner, repo) -> bool:
    rv = get_github(
        f"https://api.github.com/search/code?q=repo:{owner}/{repo}+filename:src/ontology/ontid-odk.yaml",
    )
    try:
        return 0 < rv["total_count"]
    except KeyError:
        print(f"failed to search ODK config for {owner}/{repo}:", rv)
        return False


def get_github(url, accept=None, params=None):
    headers = {
        "Authorization": f"token {TOKEN}",
    }
    if accept:
        headers["Accept"] = accept
    return requests.get(url, headers=headers, params=params).json()


def iterate_repos() -> Iterable[
    Union[tuple[str, str, str, str], tuple[str, str, None, None]]
]:
    res = requests.get(URL)
    res = yaml.safe_load(res.content)

    for record in tqdm(res["ontologies"], desc="Processing OBO conf"):
        tracker = record.get("tracker")

        # All active ontologies have trackers. Most of them
        # have GitHub, but not all. Don't consider the non-GitHub
        # ones
        if not tracker:
            tqdm.write(f'no tracker for {record["id"]}')
            yield record["id"], record["title"], None, None
        elif not tracker.startswith(PREFIX):
            tqdm.write(f'no github tracker for {record["id"]}: {tracker}')
            yield record["id"], record["title"], None, None
        else:
            # Since we assume it's a github link, slice out the prefix then
            # parse the owner and repository out of the path
            owner, repo, *_ = tracker[len(PREFIX):].split("/")
            yield record["id"], record["title"], owner, repo


def get_data(force: bool = False, test: bool = False):
    if PATH_TSV.is_file() and not force and not test:
        return pd.read_csv(PATH_TSV, sep="\t", dtype={"license": str})

    repos = sorted(iterate_repos())
    if test:
        c = 0
        repos = [
            r for r in repos if r[3] is None or (c := c + int(r[3] is not None)) <= 3
        ]
    rows = []
    repos = tqdm(repos, desc="Repositories")
    for prefix, title, owner, repo in repos:
        repos.set_postfix(repo=f"{owner}/{repo}")
        if owner is None:
            rows.append(dict(prefix=prefix, title=title, stars=0))
            continue

        # sad rate limit
        # time.sleep(1)

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
            dict(
                prefix=prefix,
                title=title,
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
                has_odk=has_odk_config(owner, repo),
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

    rows = sorted(rows, key=itemgetter("stars"), reverse=True)

    with PATH_JSON.open("w") as file:
        json.dump(rows, file, indent=2, default=str)

    # Output as an easily accessible TSV file
    df = pd.DataFrame(rows)
    df.to_csv(PATH_TSV, sep="\t", index=False)
    return rows


@click.command()
@force_option
@verbose_option
@click.option("--test", is_flag=True)
def main(force: bool, test: bool):
    rows = get_data(force=True, test=test)

    index_html = index_template.render(rows=rows)
    with INDEX.open("w") as file:
        print(index_html, file=file)

    exit(0)  # delete this when ready for ontology-specific pages
    for row in rows:
        ontology_html = ontology_template.render(row=row)
        directory = DOCS.joinpath(row["prefix"])
        directory.mkdir(exist_ok=True, parents=True)
        with directory.joinpath("index.html").open("w") as file:
            print(ontology_html, file=file)


if __name__ == "__main__":
    main()
