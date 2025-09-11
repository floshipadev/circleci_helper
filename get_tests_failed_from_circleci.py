#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pull failed Django tests from the latest failed CircleCI workflow (any branch),
and create PyCharm run configurations that run them via `manage.py test ...`.

Features:
- Loads env from ./.env (same dir as script). Does NOT override already-set env vars.
- Supports GitHub App slugs: PROJECT_SLUG=circleci/<ORG_ID>/<PROJECT_ID>
  or ORG_ID + PROJECT_ID to compose the slug.
- CIRCLECI_API_HOST override for CircleCI Server (self-hosted).
- Prefers GET /tests API; falls back to downloading JUnit XML artifacts with auth.
- Handles pagination for pipelines and tests.
- Collapses labels if the command would be too long.
- Writes BOTH: Python run-config and Django tests run-config.
- Optional grouped Django tests configs by top modules.

Required env (.env or shell):
  CIRCLECI_TOKEN=...                (personal or machine token)
  EITHER:
    PROJECT_SLUG=circleci/<ORG_ID>/<PROJECT_ID>
  OR:
    ORG_ID=<ORG_ID>
    PROJECT_ID=<PROJECT_ID>
Optional:
  BRANCH=<branch-name>              (to restrict to specific branch)
  CIRCLECI_API_HOST=https://circleci.com   (or your Server URL)
  DJANGO_SETTINGS_MODULE=...

Usage (from Django project root containing manage.py):
  python tools/circle_failed_django_to_pycharm.py
  # or with branch:
  python tools/circle_failed_django_to_pycharm.py --branch develop
  # add grouped configs (by module prefixes):
  python tools/circle_failed_django_to_pycharm.py --groups=3
  # skip Django run-config if you only want Python one:
  python tools/circle_failed_django_to_pycharm.py --no-django
"""
import os
import pathlib
import subprocess
import sys
from typing import List, Tuple, Dict, Iterable
from xml.etree import ElementTree as ET

import requests


# ------------------ .env loader ------------------
def load_env_from_dotenv(dotenv_path: pathlib.Path) -> None:
    """Load environment variables from a .env file in the same directory as the script.
    - Skips comments and blank lines
    - Supports `export KEY=VALUE` lines (shell style)
    - Strips single/double quotes
    - IMPORTANT: does NOT override already-set environment variables
    """
    if not dotenv_path.exists():
        return

    def strip_quotes(s: str) -> str:
        s = s.strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    with dotenv_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = strip_quotes(v)
            # Do not override anything that is already set in the environment
            if key and (key not in os.environ):
                os.environ[key] = val


# ------------------ helpers ------------------
def die(msg: str, hint: str = "") -> None:
    """Print an error (and optional hint) to stderr and exit with non-zero status."""
    print(msg, file=sys.stderr)
    if hint:
        print(hint, file=sys.stderr)
    sys.exit(1)


def api_base() -> str:
    """Return base URL for CircleCI v2 API, honoring CIRCLECI_API_HOST override.
    Example: https://circleci.com/api/v2
    """
    host = os.environ.get("CIRCLECI_API_HOST", "https://circleci.com").rstrip("/")
    return f"{host}/api/v2"


def get_token() -> str:
    """Fetch CircleCI API token from env; abort if missing."""
    tok = os.environ.get("CIRCLECI_TOKEN", "").strip()
    if not tok:
        die("CIRCLECI_TOKEN is not set. Specify it in .env next to the script or in the environment.")
    return tok


def resolve_project_slug() -> str:
    """Resolve project slug in the GitHub App format required by CircleCI v2 API.
    Priority:
      1) PROJECT_SLUG=circleci/<ORG_ID>/<PROJECT_ID>
      2) Compose from ORG_ID + PROJECT_ID as circleci/<ORG_ID>/<PROJECT_ID>
    """
    slug = (os.environ.get("PROJECT_SLUG") or "").strip()
    if slug:
        return slug
    org_id = (os.environ.get("ORG_ID") or "").strip()
    proj_id = (os.environ.get("PROJECT_ID") or "").strip()
    if not (org_id and proj_id):
        die("Specify PROJECT_SLUG=... or ORG_ID and PROJECT_ID in .env")
    if any(c in org_id + proj_id for c in " /"):
        die("ORG_ID/PROJECT_ID must not contain spaces. Copy ID from CircleCI UI.")
    return f"circleci/{org_id}/{proj_id}"


def http_get_json(url: str, token: str, params: Dict = None) -> Dict:
    """GET helper with Circle-Token header and JSON decoding. Raises on HTTP errors."""
    r = requests.get(url, headers={"Circle-Token": token}, params=params or {}, timeout=60)
    r.raise_for_status()
    return r.json()


def assert_project_reachable(token: str, slug: str) -> None:
    """Validate that the project exists and the token has access.
    - 404 → likely wrong slug or insufficient permissions
    - prints actionable hints for common misconfigurations
    """
    url = f"{api_base()}/project/{slug}"
    r = requests.get(url, headers={"Circle-Token": token}, timeout=60)
    if r.status_code == 404:
        die(
            f"404 Not Found for {url}",
            hint=(
                "Проверь:\n"
                "  • PROJECT_SLUG strictly kind circleci/<org-id>/<project-id> (GitHub App)\n"
                "  • Either ORG_ID/PROJECT_ID are correct\n"
                "  • Token is Personal/Machine, not Project Token\n"
            ),
        )
    r.raise_for_status()


# ------------------ CircleCI lookups ------------------
def list_pipelines_any_branch(token: str, slug: str, pages: int = 5) -> Iterable[Dict]:
    """Yield recent pipelines across any branch (paginated).
    - Each page returns up to ~50 items
    - Stops early if there is no next_page_token
    """
    url = f"{api_base()}/project/{slug}/pipeline"
    page_token = None
    for _ in range(pages):
        data = http_get_json(url, token, params={"page-token": page_token} if page_token else None)
        for it in data.get("items", []):
            yield it
        page_token = data.get("next_page_token")
        if not page_token:
            break


def latest_failed_pipeline_any_branch(token: str, slug: str) -> Tuple[str, List[Dict]]:
    """Find the most recent pipeline (any branch) that has at least one failed/error/failing workflow."""
    for p in list_pipelines_any_branch(token, slug, pages=5):
        pid = p["id"]
        wfs = http_get_json(f"{api_base()}/pipeline/{pid}/workflow", token).get("items", [])
        if any(w["status"] in ("failed", "error", "failing") for w in wfs):
            return pid, wfs
    die("No fresh pipeline with fallen workflows found (on all branches).")


def workflows_for_branch(token: str, slug: str, branch: str) -> Tuple[str, List[Dict]]:
    """Return (pipeline_id, workflows) for the most recent pipeline on a specific branch."""
    data = http_get_json(f"{api_base()}/project/{slug}/pipeline", token, params={"branch": branch})
    items = data.get("items", [])
    if not items:
        die(f"No pipelines on the branch {branch}")
    pid = items[0]["id"]
    wfs = http_get_json(f"{api_base()}/pipeline/{pid}/workflow", token).get("items", [])
    return pid, wfs


def pick_failed_workflow(workflows: List[Dict]) -> Dict:
    """Pick first workflow in a failed/error/failing state; fallback to the first if none."""
    for w in workflows:
        if w.get("status") in ("failed", "error", "failing"):
            return w
    return workflows[0] if workflows else {}


def workflow_jobs(token: str, workflow_id: str) -> List[Dict]:
    """List jobs for a given workflow."""
    return http_get_json(f"{api_base()}/workflow/{workflow_id}/job", token).get("items", [])


def list_junit_artifact_urls(token: str, slug: str, job_number: int) -> List[str]:
    """List artifact URLs (ending with .xml) for a given job. These are expected to be JUnit XML reports."""
    items = http_get_json(f"{api_base()}/project/{slug}/{job_number}/artifacts", token).get("items", [])
    return [it["url"] for it in items if it.get("url", "").endswith(".xml")]


def download_text_private(url: str, token: str) -> str:
    """Download private artifact text with Circle-Token header.
    - If 404 with header, retry by appending `?circle-token=<token>` (some endpoints require it)
    """
    r = requests.get(url, headers={"Circle-Token": token}, timeout=60)
    if r.status_code == 404:
        sep = "&" if "?" in url else "?"
        url2 = f"{url}{sep}circle-token={token}"
        r = requests.get(url2, timeout=60)
    r.raise_for_status()
    return r.text


def iter_tests_api(token: str, slug: str, job_number: int) -> Iterable[Dict]:
    """Iterate over results from the CircleCI /tests endpoint for a job, handling pagination."""
    url = f"{api_base()}/project/{slug}/{job_number}/tests"
    page_token = None
    while True:
        params = {"page-token": page_token} if page_token else None
        data = http_get_json(url, token, params=params)
        for it in data.get("items", []):
            yield it
        page_token = data.get("next_page_token")
        if not page_token:
            break


# ------------------ Mapping to Django labels ------------------
def labels_from_tests_api_items(items: Iterable[Dict]) -> List[str]:
    """Convert CircleCI /tests items into Django test labels.
    Priority:
      - classname + name → module.Class.method
      - file + name → file_path → dotted module + method
      - classname alone → module.Class
    Only include items with result in {"failure", "error"}.
    """
    out = set()
    for it in items:
        if (it.get("result") in ("failure", "error")):
            name = it.get("name") or ""
            classname = it.get("classname") or ""  # e.g. app.tests.TestClass
            file_attr = it.get("file") or ""  # path/to/test_foo.py
            label = ""
            if classname and name:
                label = f"{classname}.{name}"
            elif file_attr and name:
                dotted = file_attr.replace("/", ".").replace("\\", ".")
                if dotted.endswith(".py"):
                    dotted = dotted[:-3]
                label = f"{dotted}.{name}"
            elif classname:
                label = classname
            if label:
                out.add(label)
    return sorted(out)


def labels_from_junit_xml(xml_text: str) -> List[str]:
    """Parse JUnit XML and build Django test labels for failed/errored testcases."""
    out = set()
    root = ET.fromstring(xml_text)
    for tc in root.iter("testcase"):
        failed = any(ch.tag in ("failure", "error") for ch in tc)
        if not failed:
            continue
        name = tc.attrib.get("name") or ""
        classname = tc.attrib.get("classname") or ""
        file_attr = tc.attrib.get("file") or ""
        label = ""
        if classname and name:
            label = f"{classname}.{name}"
        elif file_attr and name:
            dotted = file_attr.replace("/", ".").replace("\\", ".")
            if dotted.endswith(".py"):
                dotted = dotted[:-3]
            label = f"{dotted}.{name}"
        elif classname:
            label = classname
        if label:
            out.add(label)
    return sorted(out)


def collapse_labels(labels: List[str]) -> List[str]:
    transformed = []
    for s in labels:
        s_list = s.split(".")
        if len(s_list) and s_list[0].startswith("test_"):
            transformed.append('.'.join(s_list[1:]))
        else:
            transformed.append(s)
    return sorted(set(transformed))


# ------------------ PyCharm run config ------------------
def detect_current_git_branch(repo_dir: pathlib.Path) -> str | None:
    """Returns the current git branch or None if it cannot be determined
    (e.g., detached HEAD or not a git repository)."""
    try:
        # fast and "quiet" way for the usual case
        res = subprocess.run(
            ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        name = res.stdout.strip()
        if res.returncode == 0 and name:
            return name
        # fallback option
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        name = res.stdout.strip()
        if res.returncode == 0 and name and name != "HEAD":  # "HEAD" means detached
            return name
    except Exception:
        pass
    return None

def find_manage_py(project_root: pathlib.Path) -> pathlib.Path:
    """Locate manage.py for the Django project.
    Search priority:
      1) <project>/src/manage.py
      2) <project>/manage.py
      3) shallow glob fallback (**/manage.py)
    """
    # priority: src/manage.py then manage.py in root
    candidates = [
        project_root / "src" / "manage.py",
        project_root / "manage.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    # as a backup option - search by tree (small, by 1 level)
    for p in project_root.glob("**/manage.py"):
        return p
    raise SystemExit("Manage.py not found (neither in root nor in src/). Specify the path manually or correct the script.")


def detect_module_name(project_root: pathlib.Path) -> str:
    """Attempt to detect PyCharm module name from .idea/modules.xml.
    Fallback to the project directory name if detection fails.
    """
    """We are trying to extract the module name from .idea/modules.xml; fallback is the name of the project folder."""
    mod_xml = project_root / ".idea" / "modules.xml"
    if mod_xml.exists():
        try:
            from xml.etree import ElementTree as ET
            tree = ET.parse(mod_xml)
            root = tree.getroot()
            for m in root.iter("module"):
                name = m.attrib.get("name")
                if name:
                    return name
        except Exception:
            pass
    return project_root.name  # backup option


def write_pycharm_python_run(project_root: pathlib.Path, labels: list[str]) -> pathlib.Path:
    """Generate a legacy 'Python' run configuration that calls manage.py directly.
    Useful for environments without the Django test runner integration enabled.
    """
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    run_dir = project_root / ".run"
    run_dir.mkdir(exist_ok=True)

    manage_path = find_manage_py(project_root)  # …/src/manage.py or …/manage.py
    working_dir = manage_path.parent
    rel_working = working_dir.relative_to(project_root).as_posix()
    rel_manage = manage_path.relative_to(project_root).as_posix()

    module_name = detect_module_name(project_root)
    params = "test " + " ".join(labels)
    dsm = os.environ.get("DJANGO_SETTINGS_MODULE", "")
    py_unbuf = "1"

    root = Element("component", {"name": "ProjectRunConfigurationManager"})
    cfg = SubElement(root, "configuration", {
        "default": "false",
        "name": "CircleCI Failed Django Tests",
        "type": "PythonConfigurationType",
        "factoryName": "Python",
    })

    # Environment / interpreter settings
    SubElement(cfg, "option", {"name": "ENV_FILES", "value": ""})
    SubElement(cfg, "option", {"name": "INTERPRETER_OPTIONS", "value": ""})
    SubElement(cfg, "option", {"name": "PARENT_ENVS", "value": "true"})

    envs = SubElement(cfg, "envs")
    SubElement(envs, "env", {"name": "PYTHONUNBUFFERED", "value": py_unbuf})
    if dsm:
        SubElement(envs, "env", {"name": "DJANGO_SETTINGS_MODULE", "value": dsm})

    # Project/module and script path
    SubElement(cfg, "option", {"name": "SDK_HOME", "value": ""})
    SubElement(cfg, "option", {"name": "WORKING_DIRECTORY", "value": f"$PROJECT_DIR$/{rel_working}"})
    SubElement(cfg, "option", {"name": "IS_MODULE_SDK", "value": "true"})
    SubElement(cfg, "option", {"name": "ADD_CONTENT_ROOTS", "value": "true"})
    SubElement(cfg, "option", {"name": "ADD_SOURCE_ROOTS", "value": "true"})
    SubElement(cfg, "module", {"name": module_name})
    SubElement(cfg, "EXTENSION", {"ID": "PythonCoverageRunConfigurationExtension", "runner": "coverage.py"})
    SubElement(cfg, "option", {"name": "SCRIPT_NAME", "value": f"$PROJECT_DIR$/{rel_manage}"})
    SubElement(cfg, "option", {"name": "PARAMETERS", "value": params})

    # UI behavior
    SubElement(cfg, "option", {"name": "SHOW_COMMAND_LINE", "value": "false"})
    SubElement(cfg, "option", {"name": "EMULATE_TERMINAL", "value": "false"})
    SubElement(cfg, "option", {"name": "MODULE_MODE", "value": "false"})
    SubElement(cfg, "option", {"name": "REDIRECT_INPUT", "value": "false"})
    SubElement(cfg, "option", {"name": "INPUT_FILE", "value": ""})
    SubElement(cfg, "method", {"v": "2"})

    cfg_path = run_dir / "CircleCI Failed Django Tests.run.xml"
    ElementTree(root).write(cfg_path, encoding="utf-8", xml_declaration=False)
    return cfg_path


def write_pycharm_django_tests_run(project_root: pathlib.Path, labels: list[str], branch: str) -> pathlib.Path:
    """
    Write a PyCharm 'Django tests' run configuration which integrates with the Test Runner UI.
    - Sets TARGET to a space-separated list of labels
    - Leaves SETTINGS/MANAGE empty so PyCharm resolves from working dir and env
    - Name includes branch in UI (but see NOTE below)

    NOTE: The filename written below is identical to the Python config's filename,
    so one may overwrite the other. If you want both files concurrently, consider
    changing the filename to e.g. 'CircleCI Failed Django Tests (Django).run.xml'.
    """
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    run_dir = project_root / ".run"
    run_dir.mkdir(exist_ok=True)

    manage_path = find_manage_py(project_root)
    working_dir = manage_path.parent
    rel_working = working_dir.relative_to(project_root).as_posix()

    module_name = detect_module_name(project_root)
    dsm = os.environ.get("DJANGO_SETTINGS_MODULE", "")

    params = " ".join(labels)

    root = Element("component", {"name": "ProjectRunConfigurationManager"})
    cfg = SubElement(root, "configuration", {
        "default": "false",
        "name": f"CircleCI Failed Django Tests ({branch or 'latest'})",
        "type": "DjangoTestsConfigurationType",
        "factoryName": "Django tests",
    })

    # Interpreter/env
    SubElement(cfg, "option", {"name": "INTERPRETER_OPTIONS", "value": ""})
    SubElement(cfg, "option", {"name": "PARENT_ENVS", "value": "true"})

    envs = SubElement(cfg, "envs")
    if dsm:
        SubElement(envs, "env", {"name": "DJANGO_SETTINGS_MODULE", "value": dsm})
    SubElement(envs, "env", {"name": "PYTHONUNBUFFERED", "value": "1"})

    # Project/module
    SubElement(cfg, "option", {"name": "WORKING_DIRECTORY", "value": f"$PROJECT_DIR$/{rel_working}"})
    SubElement(cfg, "option", {"name": "IS_MODULE_SDK", "value": "true"})
    SubElement(cfg, "option", {"name": "ADD_CONTENT_ROOTS", "value": "true"})
    SubElement(cfg, "option", {"name": "ADD_SOURCE_ROOTS", "value": "true"})
    SubElement(cfg, "module", {"name": module_name})

    # Django-specific: pass labels via TARGET
    SubElement(cfg, "option", {"name": "TARGET", "value": params})
    SubElement(cfg, "option", {"name": "USE_OPTIONS", "value": ""})
    SubElement(cfg, "option", {"name": "ADDITIONAL_OPTIONS", "value": ""})

    # Let PyCharm infer manage.py and settings
    SubElement(cfg, "option", {"name": "SETTINGS_MODULE", "value": ""})
    SubElement(cfg, "option", {"name": "CUSTOM_SETTINGS", "value": ""})
    SubElement(cfg, "option", {"name": "CUSTOM_MANAGE", "value": ""})

    SubElement(cfg, "method", {"v": "2"})

    # NOTE: Same filename as Python config above (potential overwrite).
    cfg_path = run_dir / f"CircleCI Failed Django Tests.run.xml"
    ElementTree(root).write(cfg_path, encoding="utf-8", xml_declaration=False)
    return cfg_path


def write_grouped_module_runs(project_root: pathlib.Path, labels: list[str], max_groups: int = 4) -> list[pathlib.Path]:
    """
    Split labels into up to `max_groups` buckets by module prefix and create separate Django run configs.
    Group key heuristic:
      - First two dot-separated segments (e.g., app.tests)
      - If fewer segments, use the first segment
    """
    from collections import defaultdict
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    if not labels:
        return []

    def group_key(lb: str) -> str:
        parts = lb.split(".")
        if len(parts) >= 2:
            return ".".join(parts[:2])
        return parts[0]

    buckets = defaultdict(list)
    for lb in labels:
        buckets[group_key(lb)].append(lb)

    # Sort by descending bucket size, then by name; take the top N
    groups = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_groups]

    written: list[pathlib.Path] = []
    for gname, glabels in groups:
        run_dir = project_root / ".run"
        run_dir.mkdir(exist_ok=True)

        manage_path = find_manage_py(project_root)
        working_dir = manage_path.parent
        rel_working = working_dir.relative_to(project_root).as_posix()
        module_name = detect_module_name(project_root)
        dsm = os.environ.get("DJANGO_SETTINGS_MODULE", "")

        params = " ".join(sorted(set(glabels)))

        root = Element("component", {"name": "ProjectRunConfigurationManager"})
        cfg = SubElement(root, "configuration", {
            "default": "false",
            "name": f"CircleCI Failed (Django) — {gname}",
            "type": "DjangoTestsConfigurationType",
            "factoryName": "Django tests",
        })

        # Env/interpreter
        SubElement(cfg, "option", {"name": "INTERPRETER_OPTIONS", "value": ""})
        SubElement(cfg, "option", {"name": "PARENT_ENVS", "value": "true"})
        envs = SubElement(cfg, "envs")
        if dsm:
            SubElement(envs, "env", {"name": "DJANGO_SETTINGS_MODULE", "value": dsm})
        SubElement(envs, "env", {"name": "PYTHONUNBUFFERED", "value": "1"})

        # Project/module
        SubElement(cfg, "option", {"name": "WORKING_DIRECTORY", "value": f"$PROJECT_DIR$/{rel_working}"})
        SubElement(cfg, "option", {"name": "IS_MODULE_SDK", "value": "true"})
        SubElement(cfg, "option", {"name": "ADD_CONTENT_ROOTS", "value": "true"})
        SubElement(cfg, "option", {"name": "ADD_SOURCE_ROOTS", "value": "true"})
        SubElement(cfg, "module", {"name": module_name})

        # Django target labels
        SubElement(cfg, "option", {"name": "TARGET", "value": params})
        SubElement(cfg, "option", {"name": "USE_OPTIONS", "value": ""})
        SubElement(cfg, "option", {"name": "ADDITIONAL_OPTIONS", "value": ""})
        SubElement(cfg, "option", {"name": "SETTINGS_MODULE", "value": ""})
        SubElement(cfg, "option", {"name": "CUSTOM_SETTINGS", "value": ""})
        SubElement(cfg, "option", {"name": "CUSTOM_MANAGE", "value": ""})
        SubElement(cfg, "method", {"v": "2"})

        cfg_path = run_dir / f"CircleCI Failed (Django) — {gname}.run.xml"
        ElementTree(root).write(cfg_path, encoding="utf-8", xml_declaration=False)
        written.append(cfg_path)

    return written


# ------------------ main ------------------
def main():
    """Entry point:
    - Load .env
    - Resolve token and project slug, assert access
    - Parse CLI flags/env for branch and grouping options
    - Find a failed workflow (by branch or any branch)
    - Collect failed test labels via /tests API or JUnit artifacts (fallback)
    - Collapse labels if needed to avoid overly long commands
    - Write PyCharm run configs (Python and/or Django; optional grouped)
    - Print a short summary of found labels
    """
    script_dir = pathlib.Path(__file__).resolve().parent
    load_env_from_dotenv(script_dir / ".env")

    token = get_token()
    slug = resolve_project_slug()
    assert_project_reachable(token, slug)

    # CLI flags / toggles
    branch = None
    write_django = True     # enabled by default
    write_groups = False
    groups_count = 4

    args = list(sys.argv[1:])
    if "--branch" in args:
        i = args.index("--branch")
        if i + 1 < len(args):
            branch = args[i + 1]
    if "--no-django" in args:
        write_django = False
    for a in args:
        if a == "--groups":
            write_groups = True
        elif a.startswith("--groups="):
            write_groups = True
            try:
                groups_count = max(1, int(a.split("=", 1)[1]))
            except ValueError:
                pass

    if not branch:
        branch = os.environ.get("BRANCH")

    if not branch:
        try:
            manage_path = find_manage_py(pathlib.Path(os.getcwd()))
            repo_dir = manage_path.parent
            branch = detect_current_git_branch(repo_dir)
        except SystemExit:
            pass

    # Choose workflows source: specific branch or latest any-branch failure
    if branch:
        _, workflows = workflows_for_branch(token, slug, branch)
    else:
        _, workflows = latest_failed_pipeline_any_branch(token, slug)

    if not workflows:
        die("There is no workflow in the pipeline.")
    wf = pick_failed_workflow(workflows)
    if not wf:
        die("Unable to select workflow.")

    # Collect failed test labels from each job in the workflow
    labels: List[str] = []
    for j in workflow_jobs(token, wf["id"]):
        if "job_number" not in j:
            continue  # skip approvals, etc.
        job_no = j["job_number"]

        # 1) Primary path: /tests API
        items = list(iter_tests_api(token, slug, job_no))

        failed_via_api = labels_from_tests_api_items(items)
        labels.extend(failed_via_api)


        # 2) Fallback: parse JUnit XML artifacts (if any were published)
        if not failed_via_api:
            for url in list_junit_artifact_urls(token, slug, job_no):
                try:
                    xml = download_text_private(url, token)
                    labels.extend(labels_from_junit_xml(xml))
                except requests.HTTPError:
                    # Silently continue; artifacts may not exist or be inaccessible
                    pass

    labels = sorted(set(labels))
    if not labels:
        print("No failed tests found - config was not created.")
        return

    labels = collapse_labels(labels)

    project_root = pathlib.Path(os.getcwd())
    # 1) Python run-config (back-compat / minimal integration)
    if not write_django and not write_groups:
        cfg_py = write_pycharm_python_run(project_root, labels)
        print(f"[OK] Python config: {cfg_py}")

    # 2) Django tests run-config (recommended for PyCharm test UI integration)
    if write_django:
        cfg_dj = write_pycharm_django_tests_run(project_root, labels, branch)
        print(f"[OK] Django tests config: {cfg_dj}")

    # 3) Optional: grouped Django test configs by module prefixes
    if write_groups:
        written = write_grouped_module_runs(project_root, labels, max_groups=groups_count)
        for p in written:
            print(f"[OK] Django group config: {p}")


    if branch:
        print(f"Current branch: {branch}")
    # Print a short summary (preview the first 25 labels)
    print(f"Number of failed tests found (after collapse): {len(labels)}")
    for l in labels[:25]:
        print("  ", l)
    if len(labels) > 25:
        print("  ...")


if __name__ == "__main__":
    main()
