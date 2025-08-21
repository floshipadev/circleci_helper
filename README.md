# CircleCI → PyCharm Failed Tests Runner

This tool pulls **failed Django tests** from the latest failed CircleCI workflow  
and generates **PyCharm run configurations** to re-run those tests locally via `manage.py test`.

It helps you quickly reproduce CI failures inside PyCharm without manually copying test labels.

---

## Features

- Reads CircleCI API (`/tests` endpoint, falls back to JUnit XML artifacts if needed).
- Handles pagination for pipelines and tests.
- Works across **all branches** or can be limited to a specific branch.
- Collapses test labels if too many (to avoid overly long command lines).
- Generates both:
  - **Python run-config** (`manage.py test ...` directly)
  - **Django tests run-config** (integrated with PyCharm Test Runner UI). by default.
- Optional **grouped configs** by top-level module prefix.

---

## Requirements

- Python **3.8+**
- `requests` library:
  ```bash
  pip install requests

## Usage

- Put a script in some folder, say 'tools', folder 'tools' should be on one level with your source code root project folder:
  ```text
      myproject/
        ├──src/    # (your source code)
        ├──├── manage.py
        │  └── ...
        └──tools/
           ├── get_tests_failed_from_circleci.py
           └── .env    # CircleCI token and project slug live here
  ```
  
## Examples
- Run by default (reads branch from `CIRCLE_BRANCH` env var):
    ```bash
    python tools/get_tests_failed_from_circleci.py
  
- Run for a specific branch:
    ```bash
    python tools/get_tests_failed_from_circleci.py --branch develop

- Generate grouped configs (split by module prefixes, max 3 groups):
    ```bash
    python tools/get_tests_failed_from_circleci.py --groups=3

- Skip Django run-config and only generate a Python run-config:
    ```bash
    python tools/get_tests_failed_from_circleci.py --no-django