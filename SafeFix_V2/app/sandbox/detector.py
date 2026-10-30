import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

logger = logging.getLogger("safefix.detector")

@dataclass
class ProjectEnvironmentInfo:
    """Detailed metadata about a repository's environment, build, and test requirements."""
    python_version_spec: Optional[str] = None
    dependency_manager: str = "pip"  # "uv", "poetry", "pipenv", "pip", "flit", "hatch", "pixi", "conda"
    build_system: str = "setuptools"  # "setuptools", "poetry-core", "hatchling", "flit", "meson-python", "scikit-build-core", "none"
    has_compiled_extensions: bool = False
    is_editable_supported: bool = True
    test_runner: str = "pytest"  # "pytest", "unittest", "tox", "nox"
    default_test_command: List[str] = field(default_factory=lambda: ["pytest", "-v"])
    test_discovery_command: List[str] = field(default_factory=lambda: ["pytest", "--collect-only", "-q"])
    install_commands: List[List[str]] = field(default_factory=list)
    declared_dependencies: List[str] = field(default_factory=list)
    dev_dependencies: List[str] = field(default_factory=list)
    setup_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detected_files: List[str] = field(default_factory=list)

class EnvironmentDetector:
    """Inspects a repository to determine exact build, dependency, and test execution requirements.
    
    Never assumes all Python projects can be installed with the same command.
    """

    COMPILED_EXTENSION_PATTERNS = [
        "*.pyx", "*.c", "*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp",
        "CMakeLists.txt", "meson.build", "setup.py"
    ]

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).resolve()

    def inspect(self) -> ProjectEnvironmentInfo:
        info = ProjectEnvironmentInfo()
        detected_files: List[str] = []

        # 1. Inspect Compiled Extensions
        info.has_compiled_extensions = self._check_compiled_extensions()
        if info.has_compiled_extensions:
            info.setup_notes.append("Repository contains C/C++/Cython compiled extensions or build scripts.")

        # 2. Inspect pyproject.toml
        pyproject_path = self.repo_path / "pyproject.toml"
        pyproject_data = {}
        if pyproject_path.exists():
            detected_files.append("pyproject.toml")
            pyproject_data = self._parse_pyproject(pyproject_path)
            self._apply_pyproject(info, pyproject_data)

        # 3. Inspect Lockfiles
        if (self.repo_path / "uv.lock").exists():
            detected_files.append("uv.lock")
            info.dependency_manager = "uv"
            info.setup_notes.append("Found uv.lock. Using uv for fast locked environment setup.")
        elif (self.repo_path / "poetry.lock").exists():
            detected_files.append("poetry.lock")
            info.dependency_manager = "poetry"
            info.setup_notes.append("Found poetry.lock. Using poetry dependency management.")
        elif (self.repo_path / "Pipfile.lock").exists() or (self.repo_path / "Pipfile").exists():
            f_name = "Pipfile.lock" if (self.repo_path / "Pipfile.lock").exists() else "Pipfile"
            detected_files.append(f_name)
            info.dependency_manager = "pipenv"
            info.setup_notes.append(f"Found {f_name}. Using pipenv / pip dependency resolution.")
        elif (self.repo_path / "pixi.lock").exists() or (self.repo_path / "pixi.toml").exists():
            detected_files.append("pixi.lock" if (self.repo_path / "pixi.lock").exists() else "pixi.toml")
            info.dependency_manager = "pixi"
            info.setup_notes.append("Found pixi configuration. Note: Pixi/Conda multi-language packaging.")
        elif (self.repo_path / "environment.yml").exists() or (self.repo_path / "environment.yaml").exists():
            detected_files.append("environment.yml")
            info.dependency_manager = "conda"
            info.warnings.append("Conda environment.yml found. Will use standard pip/uv fallback for Python dependencies.")

        # 4. Inspect Requirements Files
        req_files = self._find_requirements_files()
        for rf in req_files:
            detected_files.append(str(rf.relative_to(self.repo_path)))
            deps = self._parse_requirements_file(rf)
            if "test" in rf.name.lower() or "dev" in rf.name.lower():
                info.dev_dependencies.extend(deps)
            else:
                info.declared_dependencies.extend(deps)

        # 5. Inspect setup.py / setup.cfg
        if (self.repo_path / "setup.py").exists():
            detected_files.append("setup.py")
            self._inspect_setup_py(info, self.repo_path / "setup.py")
        if (self.repo_path / "setup.cfg").exists():
            detected_files.append("setup.cfg")
            self._inspect_setup_cfg(info, self.repo_path / "setup.cfg")

        # 6. Inspect Test Configurations (tox.ini, pytest.ini, noxfile.py)
        if (self.repo_path / "tox.ini").exists():
            detected_files.append("tox.ini")
            self._inspect_tox(info, self.repo_path / "tox.ini")
        if (self.repo_path / "pytest.ini").exists():
            detected_files.append("pytest.ini")
            info.test_runner = "pytest"
        if (self.repo_path / "noxfile.py").exists():
            detected_files.append("noxfile.py")
            info.setup_notes.append("Nox automation configuration detected.")

        # 7. Inspect GitHub Actions Workflows for Python versions and test commands
        self._inspect_github_workflows(info)

        # 8. Deduplicate dependencies and detect Python version
        info.declared_dependencies = sorted(list(set(info.declared_dependencies)))
        info.dev_dependencies = sorted(list(set(info.dev_dependencies)))
        info.detected_files = detected_files

        # 9. Compute Recommended Installation Commands
        info.install_commands = self._determine_install_commands(info, req_files)

        # 10. Determine Test Runner and Default Test Command
        self._determine_test_command(info)

        return info

    def _check_compiled_extensions(self) -> bool:
        """Checks if repo contains Cython/C/C++ or custom extension build logic."""
        for ext in [".pyx", ".c", ".cpp", ".cc"]:
            for _ in self.repo_path.rglob(f"*{ext}"):
                return True
        if (self.repo_path / "CMakeLists.txt").exists() or (self.repo_path / "meson.build").exists():
            return True
        setup_p = self.repo_path / "setup.py"
        if setup_p.exists():
            try:
                content = setup_p.read_text(encoding="utf-8", errors="replace")
                if "Extension(" in content or "cythonize(" in content or "scikit-build" in content:
                    return True
            except Exception:
                pass
        return False

    def _parse_pyproject(self, path: Path) -> Dict[str, Any]:
        """Safely parses pyproject.toml."""
        if not tomllib:
            return {}
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.warning(f"Could not parse pyproject.toml: {e}")
            return {}

    def _apply_pyproject(self, info: ProjectEnvironmentInfo, data: Dict[str, Any]):
        """Extracts build backend, python version, dependencies, and test runner from pyproject.toml."""
        build_sys = data.get("build-system", {})
        backend = build_sys.get("build-backend", "")
        if "poetry" in backend:
            info.build_system = "poetry-core"
            info.dependency_manager = "poetry"
        elif "hatch" in backend:
            info.build_system = "hatchling"
            info.dependency_manager = "hatch"
        elif "flit" in backend:
            info.build_system = "flit"
            info.dependency_manager = "flit"
        elif "mesonpy" in backend:
            info.build_system = "meson-python"
            info.has_compiled_extensions = True
        elif "scikit_build" in backend or "setuptools" in backend:
            info.build_system = "setuptools"

        # [project] table (PEP 621)
        project = data.get("project", {})
        if project:
            if "requires-python" in project:
                info.python_version_spec = project["requires-python"]
            if "dependencies" in project:
                for d in project["dependencies"]:
                    clean_d = self._clean_dep_name(d)
                    if clean_d:
                        info.declared_dependencies.append(clean_d)
            opt_deps = project.get("optional-dependencies", {})
            for group, deps in opt_deps.items():
                if any(kw in group.lower() for kw in ["test", "dev", "doc"]):
                    for d in deps:
                        clean_d = self._clean_dep_name(d)
                        if clean_d:
                            info.dev_dependencies.append(clean_d)

        # [tool.poetry]
        poetry = data.get("tool", {}).get("poetry", {})
        if poetry:
            deps = poetry.get("dependencies", {})
            for name in deps:
                if name.lower() == "python":
                    info.python_version_spec = str(deps[name])
                else:
                    info.declared_dependencies.append(name)
            group_dev = poetry.get("group", {}).get("dev", {}).get("dependencies", {})
            group_test = poetry.get("group", {}).get("test", {}).get("dependencies", {})
            for d in list(group_dev.keys()) + list(group_test.keys()):
                info.dev_dependencies.append(d)

        # [tool.pytest]
        if "pytest" in data.get("tool", {}):
            info.test_runner = "pytest"

    def _find_requirements_files(self) -> List[Path]:
        """Finds declared requirements files across the repository."""
        candidates = []
        for p in self.repo_path.glob("requirements*.txt"):
            candidates.append(p)
        for p in self.repo_path.glob("*-requirements.txt"):
            candidates.append(p)
        req_dir = self.repo_path / "requirements"
        if req_dir.is_dir():
            for p in req_dir.glob("*.txt"):
                candidates.append(p)
        return candidates

    def _parse_requirements_file(self, path: Path) -> List[str]:
        deps = []
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-r") or line.startswith("-e") or line.startswith("--"):
                    continue
                clean = self._clean_dep_name(line)
                if clean:
                    deps.append(clean)
        except Exception:
            pass
        return deps

    def _clean_dep_name(self, spec: str) -> str:
        """Extracts bare package name from dependency specification."""
        clean = spec.split(";")[0].strip()
        m = re.match(r"^([a-zA-Z0-9_\-\.]+)", clean)
        return m.group(1).lower() if m else ""

    def _inspect_setup_py(self, info: ProjectEnvironmentInfo, path: Path):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if "install_requires" in content:
                m = re.findall(r"['\"]([a-zA-Z0-9_\-\.]+)(?:[><=~!].*?)?['\"]", content)
                # Filter out setup parameter names
                for pkg in m:
                    if pkg not in ["install_requires", "tests_require", "setup_requires", "extras_require", "python"]:
                        info.declared_dependencies.append(pkg.lower())
        except Exception:
            pass

    def _inspect_setup_cfg(self, info: ProjectEnvironmentInfo, path: Path):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if "python_requires" in content:
                m = re.search(r"python_requires\s*=\s*(.+)", content)
                if m:
                    info.python_version_spec = m.group(1).strip()
        except Exception:
            pass

    def _inspect_tox(self, info: ProjectEnvironmentInfo, path: Path):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if "envlist" in content:
                info.setup_notes.append("Found tox.ini test matrix configuration.")
            if "pytest" in content:
                info.test_runner = "pytest"
        except Exception:
            pass

    def _inspect_github_workflows(self, info: ProjectEnvironmentInfo):
        wf_dir = self.repo_path / ".github" / "workflows"
        if not wf_dir.is_dir():
            return
        for wf in list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml")):
            try:
                content = wf.read_text(encoding="utf-8", errors="replace")
                if not info.python_version_spec:
                    m = re.search(r"python-version:\s*\[?['\"]?([0-9\.]+)['\"]?", content)
                    if m:
                        info.python_version_spec = f"~={m.group(1)}"
                        info.setup_notes.append(f"Inferred Python version {m.group(1)} from GitHub Actions workflow: {wf.name}")
                if "pytest" in content:
                    info.test_runner = "pytest"
            except Exception:
                pass

    def _determine_install_commands(self, info: ProjectEnvironmentInfo, req_files: List[Path]) -> List[List[str]]:
        """Determines ordered sequence of installation commands to prepare the environment."""
        cmds: List[List[str]] = []

        # Step B: Project Installation by Manager
        if info.dependency_manager == "uv" and (self.repo_path / "uv.lock").exists():
            cmds.append(["uv", "sync"])
        elif info.dependency_manager == "poetry" and (self.repo_path / "poetry.lock").exists():
            cmds.append(["poetry", "install", "--no-interaction", "--sync"])
            cmds.append(["uv", "sync"])
        elif info.dependency_manager == "pipenv" and (self.repo_path / "Pipfile.lock").exists():
            cmds.append(["pipenv", "install", "--dev"])
        else:
            # Install requirements files if found
            for rf in req_files:
                rel = str(rf.relative_to(self.repo_path))
                cmds.append(["pip", "install", "-r", rel])

            # Install editable package if setup.py or pyproject.toml present
            if (self.repo_path / "pyproject.toml").exists() or (self.repo_path / "setup.py").exists():
                if info.has_compiled_extensions:
                    # For packages with compiled extensions, build in-place or editable
                    cmds.append(["pip", "install", "-e", ".", "--no-build-isolation"])
                else:
                    cmds.append(["pip", "install", "-e", "."])

        # Step C: Guarantee test runner is installed
        cmds.append(["pip", "install", "pytest"])

        return cmds

    def _determine_test_command(self, info: ProjectEnvironmentInfo):
        """Determines default test and discovery commands."""
        # Find test directories
        test_dir = None
        for cand in ["tests", "test", "testing"]:
            if (self.repo_path / cand).is_dir():
                test_dir = cand
                break

        if info.test_runner == "pytest":
            if test_dir:
                info.default_test_command = ["pytest", "-v", test_dir]
                info.test_discovery_command = ["pytest", "--collect-only", "-q", test_dir]
            else:
                info.default_test_command = ["pytest", "-v"]
                info.test_discovery_command = ["pytest", "--collect-only", "-q"]
        else:
            info.default_test_command = ["python", "-m", "unittest", "discover", "-s", test_dir or "."]
            info.test_discovery_command = ["python", "-m", "unittest", "discover", "-p", "test*.py"]

