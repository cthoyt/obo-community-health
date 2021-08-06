import datetime
import pathlib
from typing import Iterable

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
TEMPLATES = HERE.joinpath('templates')
DOCS = HERE.joinpath('docs')
DOCS.mkdir(exist_ok=True, parents=True)
DATA = HERE.joinpath("data.tsv")
INDEX = DOCS.joinpath('index.html')

environment = Environment(autoescape=True, loader=FileSystemLoader(TEMPLATES), trim_blocks=False)
index_template = environment.get_template('index.html')

URL = "https://raw.githubusercontent.com/OBOFoundry/OBOFoundry.github.io/master/_config.yml"
PREFIX = "https://github.com/"
TITLE = "Add obofoundry topic to repo metadata"

# Load the GitHub access token via PyStow. We'll
# need it so we don't hit the rate limit
TOKEN = pystow.get_config("github", "token", raise_on_missing=True)

NOW = datetime.datetime.now()
ONE_YEAR_AGO = NOW - datetime.timedelta(weeks=52)


def get_most_recent_closed_issue(owner, repo):
    issues = get_issues(
        owner, repo, params={"state": "closed", "sort": "closed", "direction": "desc"}
    )
    if issues:
        return issues[0]


def get_oldest_open_issue(owner, repo):
    issues = get_issues(
        owner, repo, params={"state": "open", "sort": "updated", "direction": "asc"}
    )
    if issues:
        return issues[0]


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


def get_github(url, accept=None, params=None):
    headers = {
        "Authorization": f"token {TOKEN}",
    }
    if accept:
        headers["Accept"] = accept
    return requests.get(url, headers=headers, params=params).json()


def iterate_repos() -> Iterable[tuple[str, str]]:
    res = requests.get(URL)
    res = yaml.safe_load(res.content)

    for record in tqdm(res["ontologies"], desc="Processing OBO conf"):
        tracker = record.get("tracker")

        # All active ontologies have trackers. Most of them
        # have GitHub, but not all. Don't consider the non-GitHub
        # ones
        if not tracker or not tracker.startswith(PREFIX):
            tqdm.write(f'no tracker for {record["id"]}')
            continue

        # Since we assume it's a github link, slice out the prefix then
        # parse the owner and repository out of the path
        owner, repo, *_ = tracker[len(PREFIX):].split("/")
        yield record["id"], record["title"], owner, repo


def get_data(force: bool = False) -> pd.DataFrame:
    if DATA.is_file() and not force:
        return pd.read_csv(DATA, sep='\t', dtype={'license': str})

    repos = sorted(iterate_repos())
    rows = []
    repos = tqdm(repos, desc="Repositories")
    for prefix, title, owner, repo in repos:
        repos.set_postfix(repo=f"{owner}/{repo}")
        info = get_info(owner, repo)
        description = info["description"]
        stars = info["stargazers_count"]
        license = info["license"]
        open_issues = info["open_issues"]
        homepage = info["homepage"]
        pushed_at = info["pushed_at"]  # datetime of last push
        pushed_last_year = ONE_YEAR_AGO < dateparser.parse(info["pushed_at"]).replace(
            tzinfo=None
        )
        topics = get_topics(owner, repo)
        has_obofoundry_topic = "obofoundry" in topics
        if (
            most_recent_closed_issue := get_most_recent_closed_issue(owner, repo)
        ) is not None:
            last_close_datetime = most_recent_closed_issue["closed_at"]
            last_close_number = most_recent_closed_issue["number"]
            last_close_last_year = ONE_YEAR_AGO < dateparser.parse(
                last_close_datetime
            ).replace(tzinfo=None)
        else:
            last_close_datetime = None
            last_close_number = None
            last_close_last_year = False

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
                last_close_datetime=last_close_datetime,
                last_close_number=last_close_number,
                last_close_last_year=last_close_last_year,
            )
        )

    # Output as an easily accessible TSV file
    df = pd.DataFrame(rows).sort_values('stars', ascending=False)
    df.to_csv(DATA, sep="\t", index=False)
    return df


@click.command()
@force_option
@verbose_option
def main(force: bool):
    df = get_data(force=force)
    df.license = df.license.map(lambda x: '' if pd.isna(x) else x)

    index_html = index_template.render(df=df)
    with INDEX.open('w') as file:
        print(index_html, file=file)


if __name__ == "__main__":
    main()
