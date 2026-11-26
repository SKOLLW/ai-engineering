"""Validation pipeline for the CAG estimator: structure, configuration, security and the running service.

Usage (from the project folder):
    uv run python scripts/validate.py          # free checks only, no LLM call
    uv run python scripts/validate.py --live   # also one real estimate (costs < 0.01 USD)

Exit code 0 = all checks passed, 1 = at least one failed. Only the standard library is used.
"""

import argparse
import json
import re
import socket
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    "app/__init__.py",
    "app/main.py",
    "app/config.py",
    "app/routers/__init__.py",
    "app/routers/estimations.py",
    "app/services/__init__.py",
    "app/services/llm_service.py",
    "app/context/__init__.py",
    "app/context/examples.py",
    ".env",
    ".env.example",
    ".gitignore",
    "pyproject.toml",
    "README.md",
    "transcripts/landing_page_hubspot.json",
]
REQUIRED_DEPENDENCIES = ["fastapi", "uvicorn", "pydantic-settings", "openai", "python-dotenv"]
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "APP_ENV", "LOG_LEVEL"]
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{10,}")
SKIP_DIRS = {".venv", "__pycache__", ".git"}

SAMPLE_REQUEST = ROOT / "transcripts" / "landing_page_hubspot.json"

results: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> bool:
    results.append((ok, label))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def check_math(estimation: str, rate: int) -> tuple[bool, str]:
    """Sum of task hours == total, and total x rate == cost (same layout as the examples)."""
    tasks = [int(h) for h in re.findall(r"^\d+\. .*?: (\d+) hours", estimation, re.M)]
    total = re.search(r"Total estimate: (\d+) hours", estimation)
    cost = re.search(r"Estimated cost: ([\d,]+) USD", estimation)
    if not (tasks and total and cost):
        return False, "format not recognised (tasks, total or cost missing)"
    total_h, cost_usd = int(total.group(1)), int(cost.group(1).replace(",", ""))
    ok = sum(tasks) == total_h and total_h * rate == cost_usd
    return ok, f"tasks={sum(tasks)} h, total={total_h} h, cost={cost_usd} USD"


# --- 1. Structure ------------------------------------------------------------------------------


def check_structure() -> None:
    print("\n1. Structure")
    for rel in REQUIRED_FILES:
        check((ROOT / rel).is_file(), f"{rel} exists")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = " ".join(pyproject.get("project", {}).get("dependencies", [])).lower()
    for dep in REQUIRED_DEPENDENCIES:
        check(dep in declared, f"dependency '{dep}' in pyproject.toml")


# --- 2. Configuration and security --------------------------------------------------------------


def check_configuration_and_security() -> None:
    print("\n2. Configuration and security")
    example_env = (ROOT / ".env.example").read_text(encoding="utf-8")
    for var in REQUIRED_ENV_VARS:
        check(f"{var}=" in example_env, f".env.example documents {var}")
    check(KEY_PATTERN.search(example_env) is None, ".env.example contains no real key")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    check(".env" in (line.strip() for line in gitignore), ".env is listed in .gitignore")

    leaks = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != ".env"
        and not SKIP_DIRS.intersection(path.relative_to(ROOT).parts)
        and KEY_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    check(not leaks, "no API key in project files" + (f" (found in: {', '.join(leaks)})" if leaks else ""))

    try:
        from app.config import get_settings

        settings = get_settings()
        check(True, f"settings load from .env (provider={settings.llm_provider}, model={settings.llm_model})")
    except Exception as exc:  # noqa: BLE001 - report any settings error as a failed check
        check(False, f"settings load from .env ({type(exc).__name__})")

    from app.context.examples import ESTIMATION_EXAMPLES, HOURLY_RATE_USD

    check(len(ESTIMATION_EXAMPLES) >= 2, f"at least 2 context examples (found {len(ESTIMATION_EXAMPLES)})")
    for i, example in enumerate(ESTIMATION_EXAMPLES, start=1):
        has_keys = {"meeting_summary", "estimation"} <= example.keys()
        ok, detail = check_math(example["estimation"], HOURLY_RATE_USD) if has_keys else (False, "missing keys")
        check(has_keys and ok, f"example {i}: fields present and math correct ({detail})")


# --- 3. Running service -------------------------------------------------------------------------


def request(url: str, body: dict | None = None, timeout: float = 5) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check_service(live: bool) -> None:
    print("\n3. Running service")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        started = False
        for _ in range(40):  # up to ~20 s
            if server.poll() is not None:
                break
            try:
                started = request(f"{base}/health", timeout=1)[0] == 200
            except OSError:
                pass
            if started:
                break
            time.sleep(0.5)
        if not check(started, "server starts (uvicorn app.main:app)"):
            server.terminate()
            print("    server output:\n" + "\n".join("    " + line for line in server.communicate()[0].splitlines()[-10:]))
            return

        status, body = request(f"{base}/health")
        check(status == 200 and json.loads(body).get("status") == "ok", "GET /health -> 200, status ok")

        status, _ = request(f"{base}/docs")
        check(status == 200, "GET /docs -> 200 (Swagger)")

        paths = json.loads(request(f"{base}/openapi.json")[1]).get("paths", {})
        check("post" in paths.get("/api/v1/estimate", {}), "POST /api/v1/estimate is published in the API docs")

        status, _ = request(f"{base}/api/v1/estimate", {"transcription": ""})
        check(status == 422, "POST /api/v1/estimate with empty text -> 422")

        if not live:
            print("  [SKIP] real estimate (run with --live to include one LLM call)")
            return

        from app.context.examples import HOURLY_RATE_USD

        sample = json.loads(SAMPLE_REQUEST.read_text(encoding="utf-8"))
        status, body = request(f"{base}/api/v1/estimate", sample, timeout=90)
        if not check(status == 200, f"POST /api/v1/estimate with a transcript -> {status}"):
            return
        data = json.loads(body)
        fields = {"estimation", "model", "provider"}
        check(fields <= data.keys(), "response has estimation, model and provider")
        ok, detail = check_math(data.get("estimation", ""), HOURLY_RATE_USD)
        check(ok, f"estimate math is correct ({detail})")
        print(f"    model={data.get('model')} tokens in/out={data.get('input_tokens')}/{data.get('output_tokens')}")
    finally:
        server.terminate()
        server.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the CAG estimator project.")
    parser.add_argument("--live", action="store_true", help="also make one real LLM call (< 0.01 USD)")
    args = parser.parse_args()

    print(f"Validating {ROOT}")
    check_structure()
    check_configuration_and_security()
    check_service(args.live)

    failed = [label for ok, label in results if not ok]
    print(f"\nRESULT: {'OK' if not failed else 'FAIL'} - {len(results) - len(failed)}/{len(results)} checks passed")
    for label in failed:
        print(f"  failed: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
