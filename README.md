# obo-community-health

Check the OBO Foundry repositories to see what's going on. To run this, you'll need either the `GITHUB_TOKEN`
environment variable set up with a github token, or any other valid way to specify the `token` key in the `github`
namespace via [`pystow`](https://github.com/cthoyt/pystow). Installation and running is handled with `tox`. Run with the
following lines in your shell:

```shell
$ pip install tox
$ tox
```

For a non-standard build using a bleeding edge build of the OBO Foundry config,
use: `python build.py --force --path ~/dev/OBOFoundry.github.io/_config.yml`.

## Scores

This repository implements a highly opinionated scoring system. Here's a summary
of the distribution of scores across the OBO Foundry

![Scores](docs/score_histogram.png)
