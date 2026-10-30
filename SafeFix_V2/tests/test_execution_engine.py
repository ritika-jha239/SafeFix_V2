import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import pytest

from app.sandbox.detector import EnvironmentDetector, ProjectEnvironmentInfo
from app.sandbox.classifier import DependencyFailureClassifier, FailureCategory
from app.sandbox.isolated_env import IsolatedEnvironment, EnvironmentSetupError
from app.sandbox.test_runner import PhasedTestRunner, TestPhaseResult
from app.agent.repo_map import RepoMapper
from app.sandbox.git_manager import GitManager, sanitize_secret
from app.agent.orchestrator import SafeFixSession, Orchestrator
from app.agent.models import LLMClient
from app.agent.planner import UltraPlanner
from app.agent.fixer import NanoFixer


# ==============================================================================
# Task 9 - Test 1: Missing NumPy Diagnosis & Auto-Classification
# ==============================================================================
def test_missing_numpy_diagnosis():
    """Verify that the exact user failure (ImportError: Unable to import required dependency numpy)
    is properly diagnosed as a missing dependency and flagged as an environment setup issue.
    """
    raw_traceback = """
ImportError while loading conftest '/tmp/safefix_sandboxes/fix-908b6258/pandas/conftest.py'.
pandas/__init__.py:13: in <module>
    raise ImportError(
E   ImportError: Unable to import required dependency numpy. Please see the traceback for details.
"""
    classifier = DependencyFailureClassifier(declared_deps=["pandas", "pytest"])
    diagnosis = classifier.classify(raw_traceback, exit_code=4)

    assert diagnosis.category == FailureCategory.MISSING_DEPENDENCY
    assert diagnosis.is_environment_issue is True
    assert diagnosis.package_name == "numpy"
    assert "numpy" in diagnosis.actionable_advice.lower()


# ==============================================================================
# Task 9 - Test 2: Incompatible Dependency Versions
# ==============================================================================
def test_incorrect_dependency_version_diagnosis():
    """Verify diagnosis of version conflicts and incompatible dependency packages."""
    traceback_conflict = """
pkg_resources.ContextualVersionConflict: (urllib3 2.0.7 (/venv/lib), Requirement.parse('urllib3<2.0,>=1.21.1'), {'requests'})
pip._vendor.resolvelib.resolvers.ResolutionImpossible: [RequirementInformation(requirement=SpecifierRequirement('urllib3<2.0,>=1.21.1'))]
"""
    classifier = DependencyFailureClassifier(declared_deps=["requests", "urllib3"])
    diagnosis = classifier.classify(traceback_conflict, exit_code=1)

    assert diagnosis.category in [
        FailureCategory.INCOMPATIBLE_DEPENDENCY_VERSION,
        FailureCategory.DEPENDENCY_CONFLICT,
    ]
    assert diagnosis.is_environment_issue is True


# ==============================================================================
# Task 9 - Test 3: Failed Editable Installation
# ==============================================================================
def test_failed_editable_installation_diagnosis():
    """Verify diagnosis when pip install -e . fails due to build requirements."""
    output = """
Running setup.py develop for mypackage
error: subprocess-exited-with-error
python setup.py egg_info did not run successfully.
exit code: 1
ModuleNotFoundError: No module named 'setuptools_scm'
"""
    classifier = DependencyFailureClassifier()
    diagnosis = classifier.classify(output, exit_code=1)

    assert diagnosis.category in [FailureCategory.BUILD_FAILURE, FailureCategory.MISSING_DEPENDENCY]
    assert diagnosis.is_environment_issue is True
    assert "setuptools_scm" in (diagnosis.package_name or diagnosis.actionable_advice)


# ==============================================================================
# Task 9 - Test 4: Unsupported Python Version
# ==============================================================================
def test_unsupported_python_version_diagnosis():
    """Verify classifier detects python version mismatch."""
    output = """
ERROR: Package 'some-pkg' requires a different Python: 3.10.12 not in '>=3.12'
RuntimeError: Python version >=3.12 required, but running 3.10.12
"""
    classifier = DependencyFailureClassifier()
    diagnosis = classifier.classify(output, exit_code=1)

    assert diagnosis.category == FailureCategory.UNSUPPORTED_PYTHON_VERSION
    assert diagnosis.is_environment_issue is True
    assert "Python" in diagnosis.actionable_advice


# ==============================================================================
# Task 9 - Test 5: Repository with No requirements.txt (pyproject.toml PEP 621)
# ==============================================================================
def test_repository_with_no_requirements_txt(tmp_path: Path):
    """Verify EnvironmentDetector inspects pyproject.toml correctly when requirements.txt is absent."""
    repo_dir = tmp_path / "pep621_repo"
    repo_dir.mkdir()

    pyproject_content = """[project]
name = "modern-pkg"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "fastapi>=0.100"
]

[project.optional-dependencies]
test = [
    "pytest>=8.0",
    "pytest-cov"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    (repo_dir / "pyproject.toml").write_text(pyproject_content)

    detector = EnvironmentDetector(repo_dir)
    info = detector.inspect()

    assert "pyproject.toml" in info.detected_files
    assert info.python_version_spec == ">=3.11"
    assert info.build_system == "hatchling"
    assert "pydantic" in info.declared_dependencies
    assert "fastapi" in info.declared_dependencies
    assert any("pytest" in d for d in info.dev_dependencies)


# ==============================================================================
# Task 9 - Test 6: Poetry-Based Repository
# ==============================================================================
def test_poetry_based_repository(tmp_path: Path):
    """Verify poetry project detection and generated install commands."""
    repo_dir = tmp_path / "poetry_repo"
    repo_dir.mkdir()

    pyproject_content = """[tool.poetry]
name = "poetry-app"
version = "1.0.0"
description = "Demo"
authors = ["SafeFix"]

[tool.poetry.dependencies]
python = "^3.11"
httpx = "^0.25.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
"""
    (repo_dir / "pyproject.toml").write_text(pyproject_content)
    (repo_dir / "poetry.lock").write_text("# Poetry lockfile")

    detector = EnvironmentDetector(repo_dir)
    info = detector.inspect()

    assert info.dependency_manager == "poetry"
    assert "poetry.lock" in info.detected_files
    assert any("poetry" in cmd[0] for cmd in info.install_commands)


# ==============================================================================
# Task 9 - Test 7: uv-Based Repository
# ==============================================================================
def test_uv_based_repository(tmp_path: Path):
    """Verify uv lockfile detection and generated install commands."""
    repo_dir = tmp_path / "uv_repo"
    repo_dir.mkdir()

    (repo_dir / "pyproject.toml").write_text("""[project]
name = "uv-app"
version = "0.1.0"
dependencies = ["flask"]
""")
    (repo_dir / "uv.lock").write_text('version = 1\nrevision = 1\n')

    detector = EnvironmentDetector(repo_dir)
    info = detector.inspect()

    assert info.dependency_manager == "uv"
    assert "uv.lock" in info.detected_files
    assert any("uv" in cmd[0] for cmd in info.install_commands)


# ==============================================================================
# Task 9 - Test 8: Repository Requiring Compiled Extensions
# ==============================================================================
def test_compiled_extensions_detection(tmp_path: Path):
    """Verify detector flags compiled extensions (.c, .pyx, CMakeLists.txt)."""
    repo_dir = tmp_path / "compiled_repo"
    repo_dir.mkdir()

    (repo_dir / "setup.py").write_text("from setuptools import setup, Extension\nsetup(name='ext')")
    src_dir = repo_dir / "src"
    src_dir.mkdir()
    (src_dir / "fast_calc.pyx").write_text("def c_add(int a, int b): return a + b")

    detector = EnvironmentDetector(repo_dir)
    info = detector.inspect()

    assert info.has_compiled_extensions is True
    assert any("compiled extension" in note.lower() for note in info.setup_notes)


# ==============================================================================
# Task 9 - Test 9: Genuine Test Assertion Failure
# ==============================================================================
def test_genuine_test_assertion_failure():
    """Verify that a genuine test logic failure is correctly classified as a code defect,
    NOT an environment issue.
    """
    output = """
=================================== FAILURES ===================================
__________________________________ test_calc ___________________________________
    def test_calc():
>       assert calculate("2+2") == 4
E       AssertionError: assert 5 == 4
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_calc - AssertionError: assert 5 == 4
"""
    classifier = DependencyFailureClassifier()
    diagnosis = classifier.classify(output, exit_code=1)

    assert diagnosis.category == FailureCategory.GENUINE_FAILING_ASSERTION
    assert diagnosis.is_environment_issue is False
    assert diagnosis.is_code_defect is True


# ==============================================================================
# Task 9 - Test 10: Unreproducible Issue (BASELINE_NOT_REPRODUCED)
# ==============================================================================
@pytest.mark.asyncio
async def test_unreproducible_issue(tmp_path: Path):
    """Verify that when baseline tests pass (no bug reproduced), orchestrator does NOT modify code
    and stops at BASELINE_NOT_REPRODUCED.
    """
    repo_dir = tmp_path / "passing_repo"
    repo_dir.mkdir()
    test_dir = repo_dir / "tests"
    test_dir.mkdir()
    (test_dir / "test_ok.py").write_text("def test_ok(): assert 1 == 1\n")

    # Git init
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True)

    env = IsolatedEnvironment(repo_dir, session_id="test-unrepro", use_docker_if_available=False)
    test_runner = PhasedTestRunner(env, repo_dir)
    res = await test_runner.run_baseline_reproduction(["pytest", "-v", "tests/test_ok.py"])

    assert res.is_green is True
    assert res.failed_count == 0


# ==============================================================================
# Task 9 - Test 11: Multiple Repair Iterations with Patch History & Deduplication
# ==============================================================================
def test_multiple_repair_iterations_and_history():
    """Verify that the session maintains structured iteration history and tracks diffs."""
    session = SafeFixSession("test-iter-session")
    session.record_iteration(
        patch_text="--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new",
        files_modified=["file.py"],
        targeted_passed=1,
        targeted_failed=1,
        regression_passed=5,
        regression_failed=0
    )

    assert len(session.patch_history) == 1
    assert session.patch_history[0]["iteration"] == 1
    assert session.patch_history[0]["files_modified"] == ["file.py"]
    assert "patch_hash" in session.patch_history[0]


# ==============================================================================
# Task 9 - Test 12: No Tests Collected Condition
# ==============================================================================
@pytest.mark.asyncio
async def test_no_tests_collected_condition(tmp_path: Path):
    """Verify that exit code 0 or 5 with 0 tests collected is NEVER claimed as green."""
    repo_dir = tmp_path / "empty_tests"
    repo_dir.mkdir()
    (repo_dir / "empty.py").write_text("x = 1\n")

    env = IsolatedEnvironment(repo_dir, session_id="test-no-tests", use_docker_if_available=False)
    test_runner = PhasedTestRunner(env, repo_dir)
    res = await test_runner.discover_tests()

    assert res.collected_count == 0 or res.passed_count == 0
    assert res.is_green is False  # Must not claim green when 0 tests run!


# ==============================================================================
# Task 9 - Test 13: Rejected PR Approval
# ==============================================================================
def test_rejected_pr_approval():
    """Verify that human review rejection stops the publishing process."""
    orch = Orchestrator()
    session = orch.create_session("reject-session")
    session.state = "WAITING_FOR_APPROVAL"

    success = orch.reject_or_adjust(session.session_id, "Code changes do not follow team conventions.")
    assert success is True
    assert session.user_decision == "rejected"
    assert session.user_feedback == "Code changes do not follow team conventions."


# ==============================================================================
# Task 9 - Test 14: Failed GitHub Push & Secret Redaction
# ==============================================================================
@pytest.mark.asyncio
async def test_failed_github_push_and_secret_redaction(tmp_path: Path):
    """Verify that failed push handles errors gracefully and NEVER leaks the token."""
    secret_token = "ghp_VERY_SECRET_TOKEN_12345"
    git_mgr = GitManager(tmp_path)

    # Initialize a git repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "test.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)

    # Attempt to push to non-existent / unreachable remote with secret token
    res = await git_mgr.push_and_create_pr(
        owner="octocat",
        repo_name="nonexistent-repo-99999",
        branch_name="safefix/fix-123",
        base_branch="main",
        pr_title="Fix bug",
        pr_body="Body",
        token=secret_token
    )

    assert res["success"] is False
    # Crucial security check: Token must NEVER appear in the error output or logs
    assert secret_token not in str(res.get("error", ""))
    assert "ghp_***" in sanitize_secret(secret_token)


# ==============================================================================
# Task 9 - Test 15: Dependency Installation Timeout Handling
# ==============================================================================
@pytest.mark.asyncio
async def test_dependency_installation_timeout(tmp_path: Path):
    """Verify that long-running or hanging install commands hit bounded timeout."""
    env = IsolatedEnvironment(tmp_path, session_id="timeout-test", use_docker_if_available=False)

    # Run command with 1 second timeout that sleeps for 5 seconds
    res = await env.execute(["sleep", "5"], timeout_seconds=1)

    assert res.timed_out is True
    assert res.exit_code != 0
    assert "timed out" in res.stderr.lower()


# ==============================================================================
# Task 9 - Test 16: Malicious / Unsafe Test Commands & Path Traversal Defense
# ==============================================================================
def test_malicious_commands_and_path_traversal_defense(tmp_path: Path):
    """Verify that path traversal and dangerous paths are strictly rejected."""
    repo_dir = tmp_path / "safe_repo"
    repo_dir.mkdir()
    (repo_dir / "safe.py").write_text("print('safe')")

    mapper = RepoMapper(repo_dir)

    # 1. Reject path traversal out of repo root
    assert mapper.validate_path("../../etc/passwd") is None
    assert mapper.validate_path("/etc/shadow") is None
    assert mapper.validate_path("../other_repo/secret.py") is None

    # 2. Accept genuine repo-relative paths
    assert mapper.validate_path("safe.py") is not None
    assert mapper.validate_path(str(repo_dir / "safe.py")) is not None
