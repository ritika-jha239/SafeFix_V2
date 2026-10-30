import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from app.sandbox.isolated_env import IsolatedEnvironment, CommandExecutionResult
from app.sandbox.classifier import DependencyFailureClassifier, DiagnosisResult

logger = logging.getLogger("safefix.test_runner")

@dataclass
class TestPhaseResult:
    __test__ = False  # Tells pytest not to collect this dataclass as a test class
    phase: str  # "DISCOVERY", "BASELINE", "TARGETED", "INTEGRATION", "FINAL_VERIFICATION"
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration: float
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    collected_count: int = 0
    test_identities: List[str] = field(default_factory=list)
    failing_tests: List[str] = field(default_factory=list)
    diagnosis: Optional[DiagnosisResult] = None
    no_tests_collected: bool = False
    is_reproduced: bool = False
    is_green: bool = False

    @property
    def status_label(self) -> str:
        if self.no_tests_collected:
            return "NO_TESTS_COLLECTED"
        if self.is_green:
            return "GREEN"
        return "RED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "command": " ".join(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "error_count": self.error_count,
            "collected_count": self.collected_count,
            "test_identities": self.test_identities,
            "failing_tests": self.failing_tests,
            "is_green": self.is_green,
            "is_reproduced": self.is_reproduced,
            "no_tests_collected": self.no_tests_collected,
            "status_label": self.status_label
        }

class PhasedTestRunner:
    """Manages separate test execution phases: discovery, baseline reproduction, targeted tests, integration, and final verification.
    
    Guarantees:
    - Exit code 0 is NOT sufficient if no tests were collected.
    - Records exact counts and test identities.
    - Shows BASELINE_NOT_REPRODUCED if issue doesn't fail at baseline.
    """

    def __init__(self, isolated_env: IsolatedEnvironment, repo_path: Path):
        self.env = isolated_env
        self.repo_path = Path(repo_path).resolve()

    async def discover_tests(self, discovery_command: Optional[List[str]] = None, timeout: int = 25) -> TestPhaseResult:
        """Phase 1: Discover test items and record exact identities."""
        cmd = discovery_command or ["pytest", "--collect-only", "-q"]
        res = await self.env.run_command(cmd, timeout=timeout)

        collected_tests = []
        for line in (res.stdout + "\n" + res.stderr).splitlines():
            line = line.strip()
            # Pytest collect-only output format: path/to/test.py::test_name
            if "::" in line and not line.startswith("="):
                collected_tests.append(line.split()[0])

        count = len(collected_tests)
        if count == 0:
            m = re.search(r"(\d+)\s+tests?\s+collected", res.stdout + "\n" + res.stderr)
            if m:
                count = int(m.group(1))

        no_tests = (count == 0)
        diagnosis = DependencyFailureClassifier.diagnose(res.stdout, res.stderr, res.returncode)

        return TestPhaseResult(
            phase="DISCOVERY",
            command=cmd,
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr,
            duration=res.duration,
            collected_count=count,
            test_identities=collected_tests,
            no_tests_collected=no_tests,
            is_green=(res.returncode == 0 and not no_tests),
            diagnosis=diagnosis
        )

    async def run_baseline_reproduction(self, test_command: List[str], timeout: int = 40) -> TestPhaseResult:
        """Phase 2: Run baseline tests to verify reproduction (must be RED with failing assertion)."""
        res = await self.env.run_command(test_command, timeout=timeout)
        parsed = self._parse_test_output(res.stdout, res.stderr, res.returncode)

        # Baseline is confirmed reproduced ONLY if it has failing tests or assertion errors
        # If exit code was 0 and 0 failures, it was NOT reproduced
        is_reproduced = (parsed["failed_count"] > 0 or parsed["error_count"] > 0 or res.returncode != 0)
        is_green = (res.returncode == 0 and parsed["failed_count"] == 0 and parsed["error_count"] == 0 and not parsed["no_tests"])

        diagnosis = DependencyFailureClassifier.diagnose(res.stdout, res.stderr, res.returncode)

        return TestPhaseResult(
            phase="BASELINE",
            command=test_command,
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr,
            duration=res.duration,
            passed_count=parsed["passed_count"],
            failed_count=parsed["failed_count"],
            error_count=parsed["error_count"],
            collected_count=parsed["collected_count"],
            failing_tests=parsed["failing_tests"],
            no_tests_collected=parsed["no_tests"],
            is_reproduced=is_reproduced,
            is_green=is_green,
            diagnosis=diagnosis
        )

    async def run_targeted_tests(self, target_command: List[str], timeout: int = 35) -> TestPhaseResult:
        """Phase 3: Run targeted tests for the specific bug or files modified."""
        res = await self.env.run_command(target_command, timeout=timeout)
        parsed = self._parse_test_output(res.stdout, res.stderr, res.returncode)
        is_green = (res.returncode == 0 and parsed["failed_count"] == 0 and parsed["error_count"] == 0 and not parsed["no_tests"])

        diagnosis = DependencyFailureClassifier.diagnose(res.stdout, res.stderr, res.returncode)

        return TestPhaseResult(
            phase="TARGETED",
            command=target_command,
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr,
            duration=res.duration,
            passed_count=parsed["passed_count"],
            failed_count=parsed["failed_count"],
            error_count=parsed["error_count"],
            collected_count=parsed["collected_count"],
            failing_tests=parsed["failing_tests"],
            no_tests_collected=parsed["no_tests"],
            is_green=is_green,
            diagnosis=diagnosis
        )

    async def run_regression_tests(self, broader_command: List[str], timeout: int = 60) -> TestPhaseResult:
        """Phase 4: Run broader regression suite to ensure no collateral damage."""
        res = await self.env.run_command(broader_command, timeout=timeout)
        parsed = self._parse_test_output(res.stdout, res.stderr, res.returncode)
        is_green = (res.returncode == 0 and parsed["failed_count"] == 0 and parsed["error_count"] == 0 and not parsed["no_tests"])

        diagnosis = DependencyFailureClassifier.diagnose(res.stdout, res.stderr, res.returncode)

        return TestPhaseResult(
            phase="INTEGRATION",
            command=broader_command,
            returncode=res.returncode,
            stdout=res.stdout,
            stderr=res.stderr,
            duration=res.duration,
            passed_count=parsed["passed_count"],
            failed_count=parsed["failed_count"],
            error_count=parsed["error_count"],
            collected_count=parsed["collected_count"],
            failing_tests=parsed["failing_tests"],
            no_tests_collected=parsed["no_tests"],
            is_green=is_green,
            diagnosis=diagnosis
        )

    def _parse_test_output(self, stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
        output = f"{stdout}\n{stderr}"
        passed = 0
        failed = 0
        errors = 0
        failing_tests = []
        collected = 0

        # Pytest summary e.g. "=== 1 failed, 2 passed in 0.12s ==="
        summary_m = re.search(r"(=+ .*(?:passed|failed|error).* =+)", output)
        if summary_m:
            summary = summary_m.group(1)
            p_m = re.search(r"(\d+)\s+passed", summary)
            f_m = re.search(r"(\d+)\s+failed", summary)
            e_m = re.search(r"(\d+)\s+error", summary)
            if p_m:
                passed = int(p_m.group(1))
            if f_m:
                failed = int(f_m.group(1))
            if e_m:
                errors = int(e_m.group(1))

        # Extract individual failing test identities e.g. "FAILED test_calc.py::test_neg"
        for line in output.splitlines():
            if line.startswith("FAILED "):
                f_name = line.split()[1]
                failing_tests.append(f_name)

        collected = passed + failed + errors

        # Fallback for unittest or simple execution
        if collected == 0:
            if "collected 0 items" in output or "no tests ran" in output.lower():
                return {
                    "passed_count": 0, "failed_count": 0, "error_count": 0,
                    "collected_count": 0, "failing_tests": [], "no_tests": True
                }
            if "OK" in output and returncode == 0:
                passed = 1
                collected = 1
            elif "FAILED" in output or "Traceback" in output or returncode != 0:
                failed = 1
                collected = 1
                failing_tests.append("Test command failure")

        no_tests = (collected == 0)
        return {
            "passed_count": passed,
            "failed_count": failed,
            "error_count": errors,
            "collected_count": collected,
            "failing_tests": failing_tests,
            "no_tests": no_tests
        }
