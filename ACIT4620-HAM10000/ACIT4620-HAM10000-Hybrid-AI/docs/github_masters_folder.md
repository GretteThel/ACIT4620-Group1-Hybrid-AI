# When this project is located inside a Masters folder

There are two common setups. Choose the one that actually matches your GitHub account:

## A. `Masters` is only a folder on your computer

Put the extracted `ACIT4620-HAM10000-Hybrid-AI` project folder inside it. Create a **separate GitHub repository** called `ACIT4620-HAM10000-Hybrid-AI`. The included `.github/workflows/ci.yml` works unchanged. This is the recommended setup for a five-person team.

## B. `Masters` is a folder *inside an existing GitHub repository*

For example: `some-repo/Masters/ACIT4620-HAM10000-Hybrid-AI/`. GitHub only discovers Actions workflow YAML in the repository's top-level `.github/workflows/` directory. Move/copy the CI workflow there, and adjust its installation and test commands to run inside the project subdirectory.

Example minimal workflow file at `some-repo/.github/workflows/ham10000-ci.yml`:

```yaml
name: HAM10000 Quality Checks
on:
  push:
    paths: ['Masters/ACIT4620-HAM10000-Hybrid-AI/**', '.github/workflows/ham10000-ci.yml']
  pull_request:
    paths: ['Masters/ACIT4620-HAM10000-Hybrid-AI/**', '.github/workflows/ham10000-ci.yml']
jobs:
  verify:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: Masters/ACIT4620-HAM10000-Hybrid-AI
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m pip install -e '.[dev]'
      - run: ruff check src tests scripts
      - run: pytest
```

GitHub branch protection applies to the entire repository, not just this folder. Separate repositories make team permissions and issue tracking easier.
