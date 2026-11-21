"""The CI test runs record their timings: every test named in the log, a JUnit file per job, a job time limit.

The checks read the workflow and pyproject texts only (no yaml parser, no new dependency). Every pytest invocation
of a workflow must print each test's name (`-v`, and no `-q`), write `--junitxml=junit-*.xml`, and repeat no
`--durations` option (that pair lives in pyproject.toml `addopts`); and the job that holds an invocation must have a
`timeout-minutes` of its own and must upload `junit-*.xml` with `actions/upload-artifact` under a name with the job
and every matrix key, with `if: always()` and `continue-on-error: true` — a red run still leaves its timings, and a
failed upload never fails the job.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"


# ------------------------------------------------------------------ command lines of a workflow text

def _mask_substitutions(command: str) -> str:
    """Replace every `$(...)` with a placeholder, so its `;`, `|` and `&&` do not split the command."""
    out = []
    i = 0
    while i < len(command):
        if command.startswith("$(", i):
            depth = 0
            while i < len(command):
                if command[i] == "(":
                    depth += 1
                elif command[i] == ")":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            out.append("$SUBST")
        else:
            out.append(command[i])
            i += 1
    return "".join(out)


def _segments(command: str) -> list[str]:
    """The commands of one run line, split on `&&`, `||`, `;` and `|`."""
    parts = re.split(r"&&|\|\||;|\|", _mask_substitutions(command))
    return [p.strip() for p in parts if p.strip()]


def _strip_env_assignments(segment: str) -> str:
    while True:
        m = re.match(r'^[A-Za-z_][A-Za-z0-9_]*=("[^"]*"|\'[^\']*\'|\S+)\s+', segment)
        if not m:
            return segment
        segment = segment[m.end():]


def _tokens(command: str) -> list[str]:
    """The words of a command, with every `${{ ... }}` expression kept as one word."""
    words = command.split()
    out = []
    i = 0
    while i < len(words):
        word = words[i]
        while "${{" in word and "}}" not in word and i + 1 < len(words):
            i += 1
            word = word + " " + words[i]
        out.append(word)
        i += 1
    return out


def _pytest_command(segment: str) -> str | None:
    """The segment itself when its first word is `pytest` or `python -m pytest`, else None.

    A `pip install ... pytest` line, a `uses:` line and a comment are not invocations (their first word is not
    `pytest`), and the `$(...)` arguments of a command are not commands of their own.
    """
    text = _strip_env_assignments(segment)
    if text.startswith("#"):
        return None
    tokens = _tokens(text)
    if tokens and (tokens[0] == "pytest" or tokens[:3] == ["python", "-m", "pytest"]):
        return text
    return None


def _run_commands(lines: list[str]):
    """Yield (1-based line number, command text) for every `run:` value of the file, inline or block."""
    i = 0
    n = len(lines)
    while i < n:
        m = re.match(r"^(\s*)(?:-\s+)?run:\s*(.*)$", lines[i])
        if not m:
            i += 1
            continue
        indent = len(m.group(1))
        rest = m.group(2).strip()
        if rest and rest not in ("|", "|-", "|+", ">", ">-", ">+"):
            yield i + 1, rest
            i += 1
            continue
        block = []
        j = i + 1
        while j < n:
            raw = lines[j]
            if not raw.strip():
                block.append((j + 1, ""))
                j += 1
                continue
            if len(raw) - len(raw.lstrip()) <= indent:
                break
            block.append((j + 1, raw.strip()))
            j += 1
        k = 0
        while k < len(block):                      # join the line continuations of the block
            lineno, text = block[k]
            while text.endswith("\\") and k + 1 < len(block):
                k += 1
                text = text[:-1] + " " + block[k][1]
            if text:
                yield lineno, text
            k += 1
        i = j


def find_invocations(text: str) -> list[dict]:
    """Every pytest invocation of a workflow text: its line, its command and the job that holds it."""
    lines = text.splitlines()
    invocations = []
    for lineno, command in _run_commands(lines):
        for segment in _segments(command):
            invocation = _pytest_command(segment)
            if invocation is not None:
                invocations.append({"line": lineno, "command": invocation, "job": _owner_line(lines, lineno)})
    return invocations


# --------------------------------------------------------------------------- jobs, steps, job rules

def _indent_of(raw: str) -> int:
    return len(raw) - len(raw.lstrip())


def _job_blocks(lines: list[str]) -> list[dict]:
    """Every job under `jobs:`: its name, the 1-based line of its key, and its line range [lo, hi)."""
    jobs = []
    in_jobs = False
    jobs_indent = 0
    for i, raw in enumerate(lines):
        if not in_jobs:
            m = re.match(r"^(\s*)jobs:\s*(#.*)?$", raw)
            if m:
                in_jobs = True
                jobs_indent = len(m.group(1))
            continue
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if _indent_of(raw) == jobs_indent:
            break                                # the next top-level key ends the jobs mapping
        m = re.match(r"^(\s+)([A-Za-z_][\w-]*):\s*(.*)$", raw)
        if m and len(m.group(1)) == jobs_indent + 2:
            jobs.append({"name": m.group(2), "line": i + 1, "lo": i + 1, "hi": len(lines) + 1})
    for a, b in zip(jobs, jobs[1:]):
        a["hi"] = b["lo"]
    return jobs


def _owner_line(lines: list[str], lineno: int) -> str | None:
    for block in _job_blocks(lines):
        if block["lo"] <= lineno < block["hi"]:
            return block["name"]
    return None


def _job_timeout(body: list[tuple[int, str]], job_indent: int):
    """(line, value) of the `timeout-minutes:` at the job's own indentation, or None."""
    for lineno, raw in body:
        m = re.match(r"^(\s*)timeout-minutes:\s*(.*)$", raw)
        if m and len(m.group(1)) == job_indent + 2:
            return lineno, m.group(2).split("#")[0].strip()
    return None


def _job_matrix_keys(body: list[tuple[int, str]], job_indent: int) -> list[str]:
    """The keys of the job's `matrix:` mapping, in order."""
    for i, (lineno, raw) in enumerate(body):
        m = re.match(r"^(\s+)matrix:\s*(#.*)?$", raw)
        if m and len(m.group(1)) > job_indent:
            matrix_indent = len(m.group(1))
            keys = []
            for _lineno, raw2 in body[i + 1:]:
                if not raw2.strip() or raw2.lstrip().startswith("#"):
                    continue
                if _indent_of(raw2) <= matrix_indent:
                    break
                stripped = raw2.strip()
                m2 = re.match(r"^([A-Za-z_][\w.-]*):\s", stripped)
                if m2 and not stripped.startswith("- "):
                    keys.append(m2.group(1))
            return keys
    return []


def _scalar(value: str) -> str:
    """A step key's value without its trailing comment (a quoted value keeps its `#`)."""
    v = value.strip()
    if v[:1] in ('"', "'"):
        end = v.find(v[0], 1)
        return v[1:end] if end != -1 else v[1:]
    return v.split("#")[0].strip()


def _parse_step(body: list[tuple[int, str]], s: int, si: int) -> tuple[dict, int]:
    """One step item starting at body[s] with item indent si: its keys, its `with:` mapping, the next index."""
    step = {"line": body[s][0], "keys": {}, "with": {}}
    container = step["keys"]
    first = re.match(r"^-\s+([\w-]+):\s*(.*)$", body[s][1].strip())
    if first:
        step["keys"][first.group(1)] = (body[s][0], _scalar(first.group(2)))
    i = s + 1
    while i < len(body):
        lineno, raw = body[i]
        if not raw.strip():
            i += 1
            continue
        if _indent_of(raw) <= si:
            break
        m = re.match(r"^([\w-]+):\s*(.*)$", raw.strip())
        if m:
            if _indent_of(raw) == si + 2:
                step["keys"][m.group(1)] = (lineno, _scalar(m.group(2)))
                container = step["with"] if m.group(1) == "with" else step["keys"]
            elif _indent_of(raw) > si + 2 and container is step["with"]:
                step["with"][m.group(1)] = (lineno, _scalar(m.group(2)))
        i += 1
    return step, i


def _job_steps(body: list[tuple[int, str]], job_indent: int) -> list[dict]:
    """The steps of the job's `steps:` list."""
    for i, (lineno, raw) in enumerate(body):
        m = re.match(r"^(\s+)steps:\s*(#.*)?$", raw)
        if m and len(m.group(1)) > job_indent:
            steps_indent = len(m.group(1))
            steps = []
            j = i + 1
            while j < len(body):
                _lineno, raw2 = body[j]
                if not raw2.strip():
                    j += 1
                    continue
                if _indent_of(raw2) <= steps_indent:
                    break
                if re.match(r"^\s*-\s+", raw2):
                    step, j = _parse_step(body, j, _indent_of(raw2))
                    steps.append(step)
                    continue
                j += 1
            return steps
    return []


def _parse_workflow(text: str) -> list[dict]:
    """The jobs of a workflow text, each with its invocations, timeout, matrix keys and steps."""
    lines = text.splitlines()
    jobs = []
    for block in _job_blocks(lines):
        job_indent = _indent_of(lines[block["line"] - 1])
        body = [(i + 1, lines[i]) for i in range(block["lo"] - 1, min(block["hi"] - 1, len(lines)))]
        jobs.append({
            "name": block["name"],
            "line": block["line"],
            "lo": block["lo"],
            "hi": block["hi"],
            "timeout": _job_timeout(body, job_indent),
            "matrix_keys": _job_matrix_keys(body, job_indent),
            "steps": _job_steps(body, job_indent),
            "invocations": [],
        })
    for lineno, command in _run_commands(lines):
        for segment in _segments(command):
            invocation = _pytest_command(segment)
            if invocation is None:
                continue
            owner = next((j for j in jobs if j["lo"] <= lineno < j["hi"]), None)
            if owner is None:
                owner = {"name": None, "line": lineno, "lo": lineno, "hi": lineno + 1,
                         "timeout": None, "matrix_keys": [], "steps": [], "invocations": []}
                jobs.append(owner)
            owner["invocations"].append({"line": lineno, "command": invocation})
    return jobs


# ------------------------------------------------------------------------------- the finding functions

def _has_flag(tokens: list[str], letter: str) -> bool:
    return any(t.startswith("-") and not t.startswith("--") and letter in t[1:] for t in tokens)


def _line_findings(name: str, invocation: dict) -> list[str]:
    """The line rules of one pytest invocation; a `--collect-only` line is exempt."""
    tokens = _tokens(invocation["command"])
    if "--collect-only" in tokens:
        return []
    line = invocation["line"]
    findings = []
    junit = [t.split("=", 1)[1] for t in tokens
             if t.startswith("--junitxml=") or t.startswith("--junit-xml=")]
    if not any(v.startswith("junit-") and v.endswith(".xml") for v in junit):
        findings.append(f"{name}:{line}: the pytest invocation writes no --junitxml=junit-*.xml")
    if not (_has_flag(tokens, "v") and not _has_flag(tokens, "q") and "--quiet" not in tokens
            and "-qq" not in tokens):
        findings.append(f"{name}:{line}: the pytest invocation must print each test's name (-v) and no -q")
    if any(t.startswith("--durations") for t in tokens):
        findings.append(f"{name}:{line}: the workflow line repeats a --durations option "
                        "(the pair is held by pyproject.toml addopts)")
    return findings


def _job_findings(name: str, job: dict) -> list[str]:
    """The job rules: a timeout of its own, and an upload step that keeps a red run's timings."""
    findings = []
    job_name = job["name"]
    where = f"job '{job_name}'" if job_name is not None else "the job"
    if job["timeout"] is None:
        findings.append(f"{name}:{job['line']}: {where} runs pytest but has no timeout-minutes")
    else:
        lineno, value = job["timeout"]
        if not re.fullmatch(r"\d+", value):
            findings.append(f"{name}:{lineno}: {where} has a non-integer timeout-minutes")
    uploads = [s for s in job["steps"]
               if "uses" in s["keys"] and "actions/upload-artifact" in s["keys"]["uses"][1]]
    if not uploads:
        findings.append(f"{name}:{job['line']}: {where} runs pytest but has no actions/upload-artifact step")
    for step in uploads:
        step_line = step["line"]
        if "if" not in step["keys"] or not re.fullmatch(
                r"(\$\{\{\s*)?always\(\)(\s*\}\})?", step["keys"]["if"][1].strip()):
            findings.append(f"{name}:{step_line}: {where} upload-artifact step lacks 'if: always()'")
        if "continue-on-error" not in step["keys"] or step["keys"]["continue-on-error"][1].strip().lower() != "true":
            findings.append(f"{name}:{step_line}: {where} upload-artifact step lacks 'continue-on-error: true'")
        path_line, path_value = step["with"].get("path", (step_line, ""))
        if "junit-*.xml" not in path_value:
            findings.append(f"{name}:{path_line}: {where} upload-artifact 'path' must list junit-*.xml")
        artefact_line, artefact_value = step["with"].get("name", (step_line, ""))
        missing = []
        if not re.search(r"\$\{\{\s*github\.job\s*\}\}", artefact_value):
            missing.append("${{ github.job }}")
        for key in job["matrix_keys"]:
            if not re.search(r"\$\{\{\s*matrix\." + re.escape(key) + r"\s*\}\}", artefact_value):
                missing.append("${{ matrix." + key + " }}")
        if missing:
            findings.append(f"{name}:{artefact_line}: {where} upload-artifact 'name' lacks " + " and ".join(missing))
    return findings


def workflow_findings(text: str, name: str) -> list[str]:
    """The findings of one workflow file's text (`name` is its file name), empty when it is correct."""
    findings = []
    for job in _parse_workflow(text):
        for invocation in job["invocations"]:
            findings += _line_findings(name, invocation)
        if job["invocations"]:
            findings += _job_findings(name, job)
    return findings


def _toml_string_tokens(value: str) -> list[str]:
    v = value.strip()
    if v[:1] in ('"', "'"):
        end = v.find(v[0], 1)
        inner = v[1:end] if end != -1 else v[1:]
        return inner.split()
    return v.split()


def addopts_findings(text: str, name: str = "pyproject.toml") -> list[str]:
    """The findings of pyproject.toml's text: `addopts` under `[tool.pytest.ini_options]` holds the pair."""
    lines = text.splitlines()
    header = None
    for i, raw in enumerate(lines):
        if re.match(r"^\[tool\.pytest\.ini_options\]\s*(#.*)?$", raw):
            header = i
            break
    if header is None:
        return [f"{name}:1: no [tool.pytest.ini_options] section"]
    addopts = None
    for i in range(header + 1, len(lines)):
        if lines[i].lstrip().startswith("["):
            break
        m = re.match(r"^(\s*)addopts\s*=\s*(.*)$", lines[i])
        if m:
            addopts = (i + 1, m.group(2))
            break
    if addopts is None:
        return [f"{name}:{header + 1}: no addopts line under [tool.pytest.ini_options]"]
    lineno, raw_value = addopts
    tokens = _toml_string_tokens(raw_value)
    missing = [t for t in ("--durations=25", "--durations-min=1.0") if t not in tokens]
    if missing:
        return [f"{name}:{lineno}: addopts lacks " + " and ".join(missing)]
    return []


# ------------------------------------------------------------------------------------------- fixtures

def _workflow(job_name: str, job_lines: list[str]) -> str:
    """A workflow text with one job whose lines are given at the job's own indentation."""
    body = "\n".join(("    " + line) if line.strip() else "" for line in job_lines)
    return f"name: Tests\n\non: [pull_request]\n\njobs:\n  {job_name}:\n{body}\n"


def _line_of(text: str, needle: str) -> int:
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    raise AssertionError(f"no line contains {needle!r}")


RUN = "    run: pytest -v --junitxml=junit-unit-${{ matrix.python-version }}.xml"
UPLOAD_NAME = "      name: junit-${{ github.job }}-${{ matrix.python-version }}"
UPLOAD_STEP = [
    "  - name: Test timings (the JUnit file with every test's time)",
    "    if: always()",
    "    continue-on-error: true",
    "    uses: actions/upload-artifact@v4",
    "    with:",
    "      name: junit-${{ github.job }}-${{ matrix.python-version }}",
    "      path: junit-*.xml",
    "      if-no-files-found: warn",
    "      retention-days: 30",
]
CORRECT_UNIT = [
    "runs-on: ubuntu-latest",
    "timeout-minutes: 45  # measured maximum 27.2 min over twelve successful runs",
    "strategy:",
    "  fail-fast: false",
    "  matrix:",
    '    python-version: ["3.10", "3.12", "3.13"]',
    "steps:",
    "  - uses: actions/checkout@v4",
    "  - name: Test",
    RUN,
] + UPLOAD_STEP

RUN_EXTRA = "    run: pytest -v -rs --junitxml=junit-extra.xml"
UPLOAD_STEP_EXTRA = [
    "  - name: Test timings (the JUnit file with every test's time)",
    "    if: always()",
    "    continue-on-error: true",
    "    uses: actions/upload-artifact@v4",
    "    with:",
    "      name: junit-${{ github.job }}",
    "      path: junit-*.xml",
]
CORRECT_EXTRA = [
    "runs-on: ubuntu-latest",
    "timeout-minutes: 45  # measured maximum 27.5 min over twelve successful runs",
    "steps:",
    "  - name: Test",
    RUN_EXTRA,
] + UPLOAD_STEP_EXTRA

PYPROJECT = """\
[build-system]
requires = ["setuptools>=68", "wheel"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--durations=25 --durations-min=1.0"
"""


def _replace(lines: list[str], old: str, new: str) -> list[str]:
    out = list(lines)
    out[out.index(old)] = new
    return out


def _without(lines: list[str], old: str) -> list[str]:
    out = list(lines)
    out.remove(old)
    return out


# --------------------------------------------------------------- the planted faults, one per fixture

def test_fault_no_junitxml_is_reported_once_at_the_command_line():
    text = _workflow("unit", _replace(CORRECT_UNIT, RUN, "    run: pytest -v"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, 'run: pytest -v')}: "
                        "the pytest invocation writes no --junitxml=junit-*.xml"]


def test_fault_a_quiet_flag_is_reported_once_at_the_command_line():
    text = _workflow("unit", _replace(CORRECT_UNIT, RUN,
                                      "    run: pytest -q --junitxml=junit-unit-${{ matrix.python-version }}.xml"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, 'run: pytest -q')}: "
                        "the pytest invocation must print each test's name (-v) and no -q"]


def test_fault_quiet_and_verbose_together_are_reported_once_at_the_command_line():
    text = _workflow("unit", _replace(CORRECT_UNIT, RUN,
                                      "    run: pytest -q -v --junitxml=junit-unit-${{ matrix.python-version }}.xml"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, 'run: pytest -q -v')}: "
                        "the pytest invocation must print each test's name (-v) and no -q"]


def test_fault_a_repeated_durations_option_is_reported_once_at_the_command_line():
    text = _workflow("unit", _replace(CORRECT_UNIT, RUN, RUN + " --durations=10"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, '--durations=10')}: the workflow line repeats a "
                        "--durations option (the pair is held by pyproject.toml addopts)"]


def test_fault_a_job_without_timeout_is_reported_once_at_the_job_line():
    text = _workflow("extra", _without(CORRECT_EXTRA, "timeout-minutes: 45  # measured maximum 27.5 min over twelve successful runs"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, '  extra:')}: job 'extra' runs pytest but has no timeout-minutes"]


def test_fault_a_job_without_an_upload_step_is_reported_once_at_the_job_line():
    job = list(CORRECT_EXTRA)
    for line in UPLOAD_STEP_EXTRA:
        job.remove(line)
    text = _workflow("extra", job)
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, '  extra:')}: job 'extra' runs pytest but has no "
                        "actions/upload-artifact step"]


def test_fault_an_upload_without_always_is_reported_once_at_the_step_line():
    text = _workflow("extra", _without(CORRECT_EXTRA, "    if: always()"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, 'Test timings')}: job 'extra' upload-artifact step "
                        "lacks 'if: always()'"]


def test_fault_an_upload_without_continue_on_error_is_reported_once_at_the_step_line():
    text = _workflow("extra", _without(CORRECT_EXTRA, "    continue-on-error: true"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, 'Test timings')}: job 'extra' upload-artifact step "
                        "lacks 'continue-on-error: true'"]


def test_fault_an_upload_path_without_the_glob_is_reported_once_at_the_path_line():
    text = _workflow("extra", _replace(CORRECT_EXTRA, "      path: junit-*.xml", "      path: out.xml"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, 'path: out.xml')}: job 'extra' upload-artifact "
                        "'path' must list junit-*.xml"]


def test_fault_a_junitxml_outside_the_glob_is_reported_once_at_the_command_line():
    text = _workflow("extra", _replace(CORRECT_EXTRA, RUN_EXTRA, "    run: pytest -v --junitxml=out.xml"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, '--junitxml=out.xml')}: "
                        "the pytest invocation writes no --junitxml=junit-*.xml"]


def test_fault_a_matrix_artefact_name_without_the_matrix_key_is_reported_once_at_the_name_line():
    text = _workflow("unit", _replace(CORRECT_UNIT, UPLOAD_NAME, "      name: junit-${{ github.job }}"))
    findings = workflow_findings(text, "tests.yml")
    assert findings == [f"tests.yml:{_line_of(text, 'name: junit-${{ github.job }}')}: job 'unit' "
                        "upload-artifact 'name' lacks ${{ matrix.python-version }}"]


def test_a_correct_job_gives_no_finding():
    assert workflow_findings(_workflow("unit", CORRECT_UNIT), "tests.yml") == []
    assert workflow_findings(_workflow("extra", CORRECT_EXTRA), "tests.yml") == []


def test_a_collect_only_line_is_exempt_and_gives_no_finding():
    job = list(CORRECT_UNIT)
    job.insert(job.index(RUN) + 1, "  - name: Collect")
    job.insert(job.index(RUN) + 2, "    run: pytest --collect-only -q")
    text = _workflow("unit", job)
    assert workflow_findings(text, "tests.yml") == []
    assert len(find_invocations(text)) == 2


def test_the_invocation_rules_see_python_m_pytest_env_assignments_and_joins():
    job = [
        "runs-on: ubuntu-latest",
        "timeout-minutes: 45  # measured maximum 27.5 min over twelve successful runs",
        "steps:",
        "  - name: Install",
        "    run: |",
        "      pip install pytest",
        "      VERBOSE=1 python -m pytest -v --junitxml=junit-extra.xml \\",
        "        && python -m pytest --collect-only -q",
    ] + UPLOAD_STEP_EXTRA
    text = _workflow("extra", job)
    assert workflow_findings(text, "tests.yml") == []
    invocations = find_invocations(text)
    assert [i["command"] for i in invocations] == [
        "python -m pytest -v --junitxml=junit-extra.xml",
        "python -m pytest --collect-only -q",
    ]
    assert all(i["job"] == "extra" for i in invocations)


# ----------------------------------------------------------------------------------------- pyproject

def test_addopts_holds_the_pair():
    assert addopts_findings(PYPROJECT) == []


def test_fault_an_empty_addopts_is_reported_once_at_its_line():
    text = PYPROJECT.replace('addopts = "--durations=25 --durations-min=1.0"', 'addopts = ""')
    findings = addopts_findings(text)
    assert findings == [f"pyproject.toml:{_line_of(text, 'addopts')}: "
                        "addopts lacks --durations=25 and --durations-min=1.0"]


def test_fault_a_lost_durations_min_is_reported_once_at_the_addopts_line():
    text = PYPROJECT.replace('addopts = "--durations=25 --durations-min=1.0"', 'addopts = "--durations=25"')
    findings = addopts_findings(text)
    assert findings == [f"pyproject.toml:{_line_of(text, 'addopts')}: addopts lacks --durations-min=1.0"]


# ----------------------------------------------------------------------------------------- the real files

def test_the_real_workflows_and_addopts_hold_every_rule():
    findings = []
    invocations = 0
    for path in sorted(WORKFLOWS_DIR.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        findings += workflow_findings(text, path.name)
        invocations += len(find_invocations(text))
    findings += addopts_findings((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert findings == []
    assert invocations == 2
