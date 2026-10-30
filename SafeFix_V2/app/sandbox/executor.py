import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("safefix.executor")

# Common mapping from import module name to PyPI package name
MODULE_TO_PKG = {
    "yaml": "pyyaml",
    "dateutil": "python-dateutil",
    "cv2": "opencv-python",
    "dotenv": "python-dotenv",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "serial": "pyserial",
    "magic": "python-magic",
    "jose": "python-jose",
    "jwt": "pyjwt",
    "requests": "requests",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "pytest_mock": "pytest-mock",
    "mock": "mock",
    "httpx": "httpx",
    "jinja2": "jinja2",
    "pydantic": "pydantic",
}

class TestResult:
    def __init__(
        self,
        command: str,
        returncode: int,
        stdout: str,
        stderr: str,
        duration: float,
        passed_count: int = 0,
        failed_count: int = 0,
        error_count: int = 0,
        failures: List[str] = None
    ):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration
        self.passed_count = passed_count
        self.failed_count = failed_count
        self.error_count = error_count
        self.failures = failures or []

    @property
    def is_green(self) -> bool:
        return self.returncode == 0 and self.failed_count == 0 and self.error_count == 0

    @property
    def is_red(self) -> bool:
        return not self.is_green

    @property
    def status_label(self) -> str:
        return "GREEN" if self.is_green else "RED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": round(self.duration, 2),
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "error_count": self.error_count,
            "is_green": self.is_green,
            "status_label": self.status_label,
            "failures": self.failures
        }

class SandboxExecutor:
    """Executes tests inside a specific directory sandbox with timeout, automatic dependency installation, and output capture."""

    def __init__(self, sandbox_path: Path):
        self.sandbox_path = Path(sandbox_path)

    async def preinstall_repo_requirements(self) -> List[str]:
        """Installs test/dev requirements if present in the repository."""
        installed_files = []
        req_files = [
            "requirements-test.txt",
            "test-requirements.txt",
            "requirements-dev.txt",
            "requirements.txt"
        ]
        for rf in req_files:
            req_p = self.sandbox_path / rf
            if req_p.exists():
                logger.info(f"Pre-installing dependencies from {rf}...")
                try:
                    proc = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "pip", "install", "-r", str(req_p), "--prefer-binary", "--no-warn-script-location",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    await asyncio.wait_for(proc.communicate(), timeout=30)
                    if proc.returncode == 0:
                        logger.info(f"Successfully installed requirements from {rf}")
                        installed_files.append(rf)
                        break
                except Exception as e:
                    logger.warning(f"Skipping requirements install for {rf}: {e}")
        return installed_files

    async def auto_install_missing_dependencies(self, output: str) -> List[str]:
        """Detects missing Python dependencies from test tracebacks and installs them via pip."""
        patterns = [
            r"No module named ['\"]?([a-zA-Z0-9_\-]+)['\"]?",
            r"Unable to import required dependency ([a-zA-Z0-9_\-]+)",
            r"cannot import name .*? from ['\"]?([a-zA-Z0-9_\-]+)['\"]?",
            r"ModuleNotFoundError: ([a-zA-Z0-9_\-]+)",
            r"ImportError: ([a-zA-Z0-9_\-]+)",
        ]

        found: set = set()
        for pat in patterns:
            for m in re.finditer(pat, output):
                raw_pkg = m.group(1).strip().lower()
                # Ignore modules that belong to the local repo or python stdlib
                if (self.sandbox_path / raw_pkg).exists() or (self.sandbox_path / f"{raw_pkg}.py").exists():
                    continue
                if raw_pkg in ["sys", "os", "math", "re", "json", "typing", "collections", "itertools", "functools", "builtins", "conftest"]:
                    continue
                pkg_mapped = MODULE_TO_PKG.get(raw_pkg, raw_pkg)
                found.add(pkg_mapped)

        if not found:
            return []

        installed = []
        for pkg in found:
            logger.info(f"[SafeFix Dependency Manager] Auto-installing missing dependency: {pkg}...")
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "pip", "install", pkg, "--prefer-binary", "--no-warn-script-location",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=40)
                if proc.returncode == 0:
                    installed.append(pkg)
                    logger.info(f"[SafeFix Dependency Manager] Successfully installed: {pkg}")
                else:
                    logger.warning(f"Pip install failed for {pkg}: {stderr_b.decode('utf-8', errors='replace')[:150]}")
            except Exception as e:
                logger.warning(f"Error during auto-install of {pkg}: {e}")

        return installed

    async def run_tests(
        self,
        test_command: str = "pytest -v",
        timeout: int = 35,
        max_dependency_retries: int = 2
    ) -> TestResult:
        start_time = asyncio.get_event_loop().time()

        cmd = test_command
        if cmd.startswith("pytest"):
            python_bin = sys.executable
            cmd = f"{python_bin} -m {cmd}"

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.sandbox_path)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TERM"] = "xterm-256color"

        stdout = ""
        stderr = ""
        returncode = 0
        extra_logs = []

        for attempt in range(max_dependency_retries + 1):
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    cwd=str(self.sandbox_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout
                    )
                    stdout = stdout_bytes.decode("utf-8", errors="replace")
                    stderr = stderr_bytes.decode("utf-8", errors="replace")
                    returncode = proc.returncode or 0
                except asyncio.TimeoutError:
                    proc.kill()
                    stdout = ""
                    stderr = f"Execution timed out after {timeout} seconds."
                    returncode = -1
            except Exception as e:
                stdout = ""
                stderr = f"Failed to execute test command: {str(e)}"
                returncode = -1

            combined_output = stdout + "\n" + stderr

            # If failed, check for missing dependencies and auto-install them
            if returncode != 0 and attempt < max_dependency_retries:
                installed = await self.auto_install_missing_dependencies(combined_output)
                if installed:
                    notice = f"\n[SafeFix Dependency Manager] Auto-installed missing dependencies: {', '.join(installed)}. Re-running test suite (attempt {attempt + 2})...\n"
                    extra_logs.append(notice)
                    logger.info(notice.strip())
                    await asyncio.sleep(0.5)
                    continue

            break

        if extra_logs:
            stdout = "".join(extra_logs) + stdout

        duration = asyncio.get_event_loop().time() - start_time
        passed, failed, errors, failures = self._parse_pytest_output(stdout + "\n" + stderr)

        # Ensure returncode != 0 is never reported as 0 failures
        if returncode != 0 and passed == 0 and failed == 0 and errors == 0:
            errors = 1
            failures.append("Test command exited with non-zero status")

        return TestResult(
            command=test_command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            passed_count=passed,
            failed_count=failed,
            error_count=errors,
            failures=failures
        )

    def _parse_pytest_output(self, output: str):
        passed = 0
        failed = 0
        errors = 0
        failures = []

        # Parse pytest summary lines, e.g. "1 failed, 2 passed in 0.05s"
        # or "3 passed in 0.02s" or "1 error in 0.05s"
        summary_match = re.search(r"(=+ .*(?:passed|failed|error).* =+)", output)
        if summary_match:
            summary = summary_match.group(1)
            p_match = re.search(r"(\d+)\s+passed", summary)
            f_match = re.search(r"(\d+)\s+failed", summary)
            e_match = re.search(r"(\d+)\s+error", summary)
            if p_match:
                passed = int(p_match.group(1))
            if f_match:
                failed = int(f_match.group(1))
            if e_match:
                errors = int(e_match.group(1))

        # Also extract individual failure names e.g. "FAILED test_calculator.py::test_negatives"
        fail_lines = re.findall(r"FAILED\s+([^\s]+)", output)
        for f in fail_lines:
            failures.append(f)

        # Fallback if pytest didn't format like above (e.g. standard unittest or import failure)
        if passed == 0 and failed == 0 and errors == 0:
            if "OK" in output and "ImportError" not in output and "ModuleNotFoundError" not in output:
                passed = 1
            elif "FAILED" in output or "Traceback" in output or "ImportError" in output or "ModuleNotFoundError" in output:
                failed = 1
                failures.append("Test suite failed")

        return passed, failed, errors, failures
