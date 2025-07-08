# dev setup

1. Install:
   ```bash
   pip install -r dev_tools/requirements.txt
   ```

2. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Manual Commands
```bash
# run on all files
pre-commit run --all-files

# just formatting
black qbittensor tests --line-length 120
isort qbittensor tests

# just linting
flake8 qbittensor tests --max-line-length=120 --extend-ignore=E203,W503 --max-complexity=10

# find dead code
vulture qbittensor --min-confidence 80

# check cyclo complexity
radon cc qbittensor -a -nb  # cyclomatic complexity
radon mi qbittensor -nb      # maintainability index
```

## Skip Pre-commit

```bash
git commit --no-verify
```
