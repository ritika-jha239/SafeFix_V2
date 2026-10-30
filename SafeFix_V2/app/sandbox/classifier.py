import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

class FailureCategory(str, Enum):
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    INCOMPATIBLE_DEPENDENCY_VERSION = "INCOMPATIBLE_DEPENDENCY_VERSION"
    UNSUPPORTED_PYTHON_VERSION = "UNSUPPORTED_PYTHON_VERSION"
    MISSING_COMPILED_EXTENSION = "MISSING_COMPILED_EXTENSION"
    MISSING_SYSTEM_DEPENDENCY = "MISSING_SYSTEM_DEPENDENCY"
    BUILD_FAILURE = "BUILD_FAILURE"
    NETWORK_OR_PACKAGE_INDEX_FAILURE = "NETWORK_OR_PACKAGE_INDEX_FAILURE"
    DEPENDENCY_CONFLICT = "DEPENDENCY_CONFLICT"
    TEST_COLLECTION_FAILURE = "TEST_COLLECTION_FAILURE"
    GENUINE_FAILING_ASSERTION = "GENUINE_FAILING_ASSERTION"
    TEST_EXECUTION_TIMEOUT = "TEST_EXECUTION_TIMEOUT"
    NO_TESTS_COLLECTED = "NO_TESTS_COLLECTED"
    COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
    UNKNOWN_FAILURE = "UNKNOWN_FAILURE"

@dataclass
class DiagnosisResult:
    category: FailureCategory
    is_environment_issue: bool
    is_code_bug: bool
    actionable_message: str
    missing_packages: List[str] = field(default_factory=list)
    missing_extensions: List[str] = field(default_factory=list)
    system_dependencies: List[str] = field(default_factory=list)
    suggested_repair_command: Optional[List[str]] = None

    @property
    def package_name(self) -> Optional[str]:
        return self.missing_packages[0] if self.missing_packages else None

    @property
    def actionable_advice(self) -> str:
        return self.actionable_message

    @property
    def is_code_defect(self) -> bool:
        return self.is_code_bug

# Known mapping of Python import names to installable PyPI distributions
IMPORT_TO_PYPI = {
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
    "cryptography": "cryptography",
    "urllib3": "urllib3",
}

class DependencyFailureClassifier:
    """Classifies execution, build, and test failures to distinguish environment issues from code bugs."""

    def __init__(self, declared_deps: Optional[List[str]] = None):
        self.declared_deps = declared_deps or []

    def classify(self, output: str, exit_code: int = 1) -> DiagnosisResult:
        return self.diagnose(
            stdout=output,
            stderr="",
            returncode=exit_code,
            declared_dependencies=self.declared_deps
        )

    @classmethod
    def diagnose(
        cls,
        stdout: str,
        stderr: str,
        returncode: int,
        declared_dependencies: Optional[List[str]] = None
    ) -> DiagnosisResult:
        output = f"{stdout}\n{stderr}"
        declared = set(d.lower() for d in (declared_dependencies or []))

        # 1. Test Execution Timeout
        if "Execution timed out" in output or returncode == 124 or "timed out after" in output:
            return DiagnosisResult(
                category=FailureCategory.TEST_EXECUTION_TIMEOUT,
                is_environment_issue=False,
                is_code_bug=True,
                actionable_message="Execution timed out. The test or patch may have introduced an infinite loop or deadlock."
            )

        # 2. Command not found
        if "command not found" in output or "No such file or directory: '" in output or returncode == 127:
            m = re.search(r"([a-zA-Z0-9_\-]+):\s*command not found", output)
            cmd = m.group(1) if m else "Executable"
            return DiagnosisResult(
                category=FailureCategory.COMMAND_NOT_FOUND,
                is_environment_issue=True,
                is_code_bug=False,
                actionable_message=f"Required command '{cmd}' is not found in the environment."
            )

        # 3. Unsupported Python Version
        if any(term in output for term in [
            "Requires-Python",
            "requires a different Python",
            "Python version not supported",
            "unsupported Python version",
            "is not compatible with Python",
        ]):
            m = re.search(r"requires (?:a different )?Python\s*([<>=!~0-9\.]+)", output)
            spec = m.group(1) if m else "different version"
            return DiagnosisResult(
                category=FailureCategory.UNSUPPORTED_PYTHON_VERSION,
                is_environment_issue=True,
                is_code_bug=False,
                actionable_message=f"Unsupported Python version. Project requirements: {spec}."
            )

        # 4. Missing Compiled Extension / Native Module Build Issue
        if any(term in output for term in [
            "No module named '_libs'",
            "cannot import name '_",
            "compiled against a different version of",
            "Cython extension not built",
            "extension was not compiled",
            "did you run setup.py build_ext --inplace",
            "meson.build: error",
            "C extension not loaded",
        ]):
            m = re.search(r"cannot import name ['\"](_[a-zA-Z0-9_\-]+)['\"]", output)
            ext_name = m.group(1) if m else "native extension"
            return DiagnosisResult(
                category=FailureCategory.MISSING_COMPILED_EXTENSION,
                is_environment_issue=True,
                is_code_bug=False,
                missing_extensions=[ext_name],
                actionable_message=f"Missing compiled extension '{ext_name}'. The package requires native C/Cython compilation before running tests.",
                suggested_repair_command=["python", "setup.py", "build_ext", "--inplace"]
            )

        # 5. Missing System Dependency / Header
        if any(term in output for term in [
            "fatal error: Python.h: No such file",
            "error: command 'gcc' failed",
            "error: command 'clang' failed",
            "libffi.so",
            "libssl.so",
            "cannot open shared object file: No such file",
            "cmake: not found",
            "pkg-config: not found",
        ]):
            m = re.search(r"fatal error:\s*([a-zA-Z0-9_\-\.\/]+\.h):\s*No such file", output)
            header = m.group(1) if m else "native library"
            return DiagnosisResult(
                category=FailureCategory.MISSING_SYSTEM_DEPENDENCY,
                is_environment_issue=True,
                is_code_bug=False,
                system_dependencies=[header],
                actionable_message=f"Missing system compiler or header dependency: {header}."
            )

        # 6. Network or Package Index Failure
        if any(term in output for term in [
            "Could not find a version that satisfies the requirement",
            "Connection refused",
            "Failed to establish a new connection",
            "HTTPError: 503",
            "HTTPError: 403",
            "ReadTimeoutError",
            "Temporary failure in name resolution",
        ]):
            return DiagnosisResult(
                category=FailureCategory.NETWORK_OR_PACKAGE_INDEX_FAILURE,
                is_environment_issue=True,
                is_code_bug=False,
                actionable_message="Network or PyPI package index failure during dependency retrieval."
            )

        # 7. Incompatible Dependency Version / Conflict
        if any(term in output for term in [
            "VersionConflict",
            "ResolutionImpossible",
            "ContextualVersionConflict",
            "has requirement",
            "which is incompatible with",
            "conflicting dependencies",
        ]):
            return DiagnosisResult(
                category=FailureCategory.INCOMPATIBLE_DEPENDENCY_VERSION,
                is_environment_issue=True,
                is_code_bug=False,
                actionable_message="Dependency version conflict detected. Incompatible version installed or requested."
            )

        # 8. Build Failure (pip wheel / setup.py / pyproject)
        if any(term in output for term in [
            "Failed building wheel for",
            "error: subprocess-exited-with-error",
            "setup.py bdist_wheel failed",
            "flit_core.buildapi",
            "BackendUnavailable",
        ]):
            # Check if there is an underlying ModuleNotFoundError / missing build requirement
            m_mod = re.search(r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]?([a-zA-Z0-9_\-]+)['\"]?", output)
            missing = []
            if m_mod:
                mod = m_mod.group(1)
                pkg = IMPORT_TO_PYPI.get(mod, mod)
                missing = [pkg]
            m = re.search(r"Failed building wheel for\s+([a-zA-Z0-9_\-\.]+)", output)
            pkg = m.group(1) if m else "package"
            msg = f"Build failed for package '{pkg}'."
            if missing:
                msg += f" Missing build requirement: {missing[0]}."
            return DiagnosisResult(
                category=FailureCategory.BUILD_FAILURE,
                is_environment_issue=True,
                is_code_bug=False,
                actionable_message=msg,
                missing_packages=missing,
                suggested_repair_command=["pip", "install", missing[0]] if missing else None
            )

        # 9. Missing Dependency (ModuleNotFoundError or ImportError)
        missing_pkgs: Set[str] = set()
        dep_patterns = [
            r"ModuleNotFoundError:\s*No module named ['\"]?([a-zA-Z0-9_\-]+)['\"]?",
            r"ImportError:\s*No module named ['\"]?([a-zA-Z0-9_\-]+)['\"]?",
            r"Unable to import required dependency\s+([a-zA-Z0-9_\-]+)",
            r"cannot import name .*? from ['\"]?([a-zA-Z0-9_\-]+)['\"]?",
        ]
        for pat in dep_patterns:
            for match in re.finditer(pat, output):
                raw = match.group(1).strip().lower()
                if raw not in ["sys", "os", "math", "re", "json", "typing", "collections", "itertools", "functools", "builtins"]:
                    pypi_name = IMPORT_TO_PYPI.get(raw, raw)
                    missing_pkgs.add(pypi_name)

        if missing_pkgs:
            pkgs_list = sorted(list(missing_pkgs))
            # Cross-check declared project requirements
            matching_declared = [p for p in pkgs_list if p in declared]
            actionable = f"Missing Python dependency: {', '.join(pkgs_list)}."
            if matching_declared:
                actionable += f" (Declared in project requirements: {', '.join(matching_declared)})"
            
            return DiagnosisResult(
                category=FailureCategory.MISSING_DEPENDENCY,
                is_environment_issue=True,
                is_code_bug=False,
                missing_packages=pkgs_list,
                actionable_message=actionable,
                suggested_repair_command=["pip", "install"] + pkgs_list
            )

        # 10. Test Collection Failure
        if "Interrupted: 1 error during collection" in output or "errors during collection" in output or "ImportError while loading conftest" in output:
            return DiagnosisResult(
                category=FailureCategory.TEST_COLLECTION_FAILURE,
                is_environment_issue=True,
                is_code_bug=False,
                actionable_message="Test runner failed during test collection phase before any tests could execute."
            )

        # 11. No tests collected
        if "collected 0 items" in output or "no tests ran" in output.lower():
            return DiagnosisResult(
                category=FailureCategory.NO_TESTS_COLLECTED,
                is_environment_issue=True,
                is_code_bug=False,
                actionable_message="No tests were collected. Test command exit code is non-conclusive."
            )

        # 12. Genuine Failing Assertion / Code Bug
        if any(term in output for term in [
            "AssertionError",
            "assert ",
            "FAILED ",
            "pytest.raises",
            "ValueError:",
            "TypeError:",
            "KeyError:",
            "AttributeError:",
            "IndexError:",
        ]) and returncode != 0:
            return DiagnosisResult(
                category=FailureCategory.GENUINE_FAILING_ASSERTION,
                is_environment_issue=False,
                is_code_bug=True,
                actionable_message="Genuine code test failure or assertion error reproduced in sandbox."
            )

        # 13. Default Unknown
        clean_err = (stderr.strip() or stdout.strip())
        last_lines = "\n".join(clean_err.splitlines()[-6:]) if clean_err else f"Process exited with code {returncode}."
        err_msg = f"Process exited with code {returncode}:\n{last_lines}" if returncode != 0 else "Execution completed successfully."
        return DiagnosisResult(
            category=FailureCategory.UNKNOWN_FAILURE if returncode != 0 else FailureCategory.GENUINE_FAILING_ASSERTION,
            is_environment_issue=(returncode != 0),
            is_code_bug=False,
            actionable_message=err_msg
        )
