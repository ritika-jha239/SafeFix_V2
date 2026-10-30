import asyncio
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.sandbox.detector import ProjectEnvironmentInfo
from app.sandbox.classifier import DependencyFailureClassifier, DiagnosisResult, FailureCategory

logger = logging.getLogger("safefix.isolated_env")

@dataclass
class CommandExecutionResult:
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def exit_code(self) -> int:
        return self.returncode

    @property
    def is_success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

class EnvironmentSetupError(Exception):
    """Raised when environment installation or setup cannot finish."""
    def __init__(self, message: str, diagnosis: DiagnosisResult, command: List[str], stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.message = message
        self.diagnosis = diagnosis
        self.command = command
        self.stdout = stdout
        self.stderr = stderr

class IsolatedEnvironment:
    """Manages an isolated execution environment for a repair session.
    
    Guarantees:
    - Does NOT install repository dependencies into SafeFix application interpreter.
    - Uses disposable container when Docker is available (with CPU, memory, timeout limits),
      or session-scoped dedicated virtualenv ({workspace}/.venv) with uv/pip.
    - Every command has timeout, exit-code verification, captured output, and structured logs.
    - Stops with ENVIRONMENT_FAILED on unresolvable setup failures.
    """

    def __init__(
        self,
        workspace_path: Path,
        session_id: str,
        use_docker_if_available: bool = True,
        cpu_limit: str = "2.0",
        memory_limit: str = "2g",
        docker_image: str = "safefix-runner:local"
    ):
        self.workspace_path = Path(workspace_path).resolve()
        self.session_id = session_id
        self.use_docker_if_available = use_docker_if_available
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.docker_image = docker_image

        self.venv_path = self.workspace_path / ".venv"
        self.venv_bin = self.venv_path / "bin"
        self.container_name = f"safefix-env-{self.session_id}"
        self.is_docker = False
        self.is_ready = False
        self.structured_logs: List[Dict[str, Any]] = []

    async def ensure_ready(self) -> str:
        return await self.initialize()

    async def initialize(self) -> str:
        """Determines isolation mode and initializes the fresh environment."""
        if self.is_ready:
            return "docker" if self.is_docker else "venv"

        # Check if Docker is available and runnable
        if self.use_docker_if_available and shutil.which("docker"):
            try:
                # Remove any existing container with this name
                rm_proc = await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", self.container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await rm_proc.wait()

                # Start a persistent session container with host networking for reliable PyPI access and uv support
                uv_host_path = shutil.which("uv") or "/home/ritika/.local/bin/uv"
                uv_mount = ["-v", f"{uv_host_path}:/usr/local/bin/uv:ro"] if os.path.exists(uv_host_path) else []
                docker_args = [
                    "docker", "run", "-d",
                    "--name", self.container_name,
                    f"--cpus={self.cpu_limit}",
                    f"--memory={self.memory_limit}",
                    "--network=host",
                    "-v", f"{self.workspace_path}:/workspace:rw",
                    "-w", "/workspace",
                ] + uv_mount + [
                    self.docker_image,
                    "sleep", "infinity"
                ]
                proc = await asyncio.create_subprocess_exec(
                    *docker_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=10.0)
                if proc.returncode == 0:
                    self.is_docker = True
                    self.is_ready = True
                    logger.info(f"Session {self.session_id}: Using persistent Docker container {self.container_name} ({self.docker_image})")
                    # Configure git safe directory in container
                    await self.run_command(["git", "config", "--global", "--add", "safe.directory", "*"])
                    # Outbound HTTPS / DNS connectivity check to PyPI inside the container
                    net_check = await self.run_command(
                        ["python", "-c", "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/', timeout=5)"],
                        timeout=8
                    )
                    if net_check.is_success:
                        logger.info(f"Session {self.session_id}: Outbound PyPI network connectivity verified.")
                    else:
                        logger.warning(f"Session {self.session_id}: Outbound PyPI network issue: {net_check.stderr}")
                    return "docker"
            except Exception as e:
                logger.info(f"Docker not runnable, falling back to dedicated venv isolation: {e}")

        # Fallback to Dedicated Virtualenv Isolation (NEVER host interpreter)
        self.is_docker = False
        logger.info(f"Session {self.session_id}: Using session-specific virtualenv isolation (reduced isolation: Docker unavailable)")
        await self._create_dedicated_venv()
        self.is_ready = True
        return "venv"

    async def _create_dedicated_venv(self):
        """Creates a completely isolated virtual environment for this session."""
        if self.venv_path.exists():
            shutil.rmtree(self.venv_path, ignore_errors=True)

        uv_path = shutil.which("uv") or "/home/ritika/.local/bin/uv"
        if os.path.exists(uv_path):
            # uv creates venv in ~50ms
            proc = await asyncio.create_subprocess_exec(
                uv_path, "venv", str(self.venv_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
        else:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "venv", str(self.venv_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)

        if not (self.venv_bin / "python").exists():
            raise RuntimeError(f"Failed to create isolated virtualenv at {self.venv_path}")

        # Pre-install pytest inside the session virtualenv so tests never need host fallback
        try:
            if os.path.exists(uv_path):
                p_inst = await asyncio.create_subprocess_exec(
                    uv_path, "pip", "install", "pytest", "--python", str(self.venv_bin / "python"),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.wait_for(p_inst.communicate(), timeout=20)
            else:
                p_inst = await asyncio.create_subprocess_exec(
                    str(self.venv_bin / "python"), "-m", "pip", "install", "pytest",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.wait_for(p_inst.communicate(), timeout=35)
        except Exception as e:
            logger.warning(f"Could not pre-install pytest into venv: {e}")

    async def cleanup(self):
        """Terminates persistent container or clears virtualenv."""
        if self.is_docker:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", self.container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await asyncio.wait_for(proc.communicate(), timeout=5)
            except Exception:
                pass

    async def run_command(
        self,
        command: List[str],
        timeout: int = 40,
        env_vars: Optional[Dict[str, str]] = None,
        cwd: Optional[Path] = None,
        network: str = "bridge"
    ) -> CommandExecutionResult:
        """Executes a command safely with timeout, exit verification, output capture, and resource limits.
        
        Never runs unsafe arbitrary shell strings.
        """
        start_time = time.time()
        working_dir = cwd or self.workspace_path
        timed_out = False

        if not self.is_ready:
            await self.ensure_ready()

        # Persistent Docker Container Execution (Setup and tests share the exact same container)
        if self.is_docker:
            rel_dir = f"/workspace/{working_dir.relative_to(self.workspace_path)}" if working_dir != self.workspace_path else "/workspace"
            docker_cmd = [
                "docker", "exec",
                "-w", rel_dir,
                "-e", "PYTHONPATH=/workspace",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "UV_LINK_MODE=copy",
                "-e", "SETUPTOOLS_SCM_PRETEND_VERSION=9.0.0.dev0",
                "-e", "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST=9.0.0.dev0",
                "-e", "GIT_DIR=/workspace/.git",
                "-e", "GIT_WORK_TREE=/workspace",
                "-e", "TERM=xterm-256color"
            ]
            if env_vars:
                for k, v in env_vars.items():
                    docker_cmd.extend(["-e", f"{k}={v}"])
            
            docker_cmd.append(self.container_name)
            docker_cmd.extend(command)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    stdout = stdout_b.decode("utf-8", errors="replace")
                    stderr = stderr_b.decode("utf-8", errors="replace")
                    returncode = proc.returncode or 0
                except asyncio.TimeoutError:
                    proc.kill()
                    timed_out = True
                    stdout = ""
                    stderr = f"Command timed out after {timeout} seconds."
                    returncode = 124
            except Exception as e:
                stdout = ""
                stderr = f"Docker execution error: {e}"
                returncode = -1

        # Dedicated Virtualenv Execution (Per-session sandbox venv, strictly isolated from host)
        else:
            resolved_cmd = list(command)
            bin_name = resolved_cmd[0]

            # Re-route python, pip, pytest, uv to sandbox's dedicated virtualenv
            if bin_name in ["python", "python3"]:
                resolved_cmd[0] = str(self.venv_bin / "python")
            elif bin_name == "pip":
                resolved_cmd = [str(self.venv_bin / "python"), "-m", "pip"] + resolved_cmd[1:]
            elif bin_name == "pytest":
                if (self.venv_bin / "pytest").exists():
                    resolved_cmd = [str(self.venv_bin / "pytest")] + resolved_cmd[1:]
                elif (self.venv_bin / "python").exists():
                    resolved_cmd = [str(self.venv_bin / "python"), "-m", "pytest"] + resolved_cmd[1:]
                else:
                    return CommandExecutionResult(
                        command=command,
                        returncode=127,
                        stdout="",
                        stderr="ENVIRONMENT_FAILED: pytest is not installed inside the session isolated environment.",
                        duration=0.0
                    )

            env = os.environ.copy()
            env["VIRTUAL_ENV"] = str(self.venv_path)
            env["PATH"] = f"{self.venv_bin}:{env.get('PATH', '')}"
            env["PYTHONPATH"] = str(self.workspace_path)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["UV_LINK_MODE"] = "copy"
            if env_vars:
                env.update(env_vars)

            try:
                proc = await asyncio.create_subprocess_exec(
                    *resolved_cmd,
                    cwd=str(working_dir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    stdout = stdout_b.decode("utf-8", errors="replace")
                    stderr = stderr_b.decode("utf-8", errors="replace")
                    returncode = proc.returncode or 0
                except asyncio.TimeoutError:
                    proc.kill()
                    timed_out = True
                    stdout = ""
                    stderr = f"Command timed out after {timeout} seconds."
                    returncode = 124
            except Exception as e:
                stdout = ""
                stderr = f"Execution error: {e}"
                returncode = -1

        duration = time.time() - start_time
        res = CommandExecutionResult(
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration=round(duration, 2),
            timed_out=timed_out
        )

        # Structured logging
        log_entry = {
            "time": time.time(),
            "command": " ".join(command),
            "returncode": returncode,
            "duration": res.duration,
            "timed_out": timed_out,
            "stdout_preview": stdout[:200] if stdout else "",
            "stderr_preview": stderr[:200] if stderr else ""
        }
        self.structured_logs.append(log_entry)

        return res

    async def execute(
        self,
        command: List[str],
        timeout_seconds: int = 40,
        env_vars: Optional[Dict[str, str]] = None,
        cwd: Optional[Path] = None,
        network: str = "bridge"
    ) -> CommandExecutionResult:
        """Alias for run_command with timeout_seconds argument."""
        return await self.run_command(command, timeout=timeout_seconds, env_vars=env_vars, cwd=cwd, network=network)

    async def setup(self, env_info: ProjectEnvironmentInfo, max_repair_retries: int = 2) -> List[CommandExecutionResult]:
        """Executes environment setup commands with failure classification and bounded retries.
        
        Raises EnvironmentSetupError if setup cannot finish.
        """
        results: List[CommandExecutionResult] = []
        install_cmds = list(env_info.install_commands)

        for cmd in install_cmds:
            res = await self.run_command(cmd, timeout=90)
            results.append(res)

            if not res.is_success:
                # Classify the installation failure
                diag = DependencyFailureClassifier.diagnose(
                    stdout=res.stdout,
                    stderr=res.stderr,
                    returncode=res.returncode,
                    declared_dependencies=env_info.declared_dependencies
                )

                # Attempt repair if fixable (e.g. missing declared dependency)
                repaired = False
                if diag.is_environment_issue and diag.suggested_repair_command and max_repair_retries > 0:
                    logger.info(f"Attempting environment repair: {' '.join(diag.suggested_repair_command)}")
                    repair_res = await self.run_command(diag.suggested_repair_command, timeout=60)
                    results.append(repair_res)
                    if repair_res.is_success:
                        # Re-run original command
                        retry_res = await self.run_command(cmd, timeout=90)
                        results.append(retry_res)
                        if retry_res.is_success:
                            repaired = True

                if not repaired:
                    clean_err = (res.stderr.strip() or res.stdout.strip())
                    last_err = "\n".join(clean_err.splitlines()[-8:]) if clean_err else ""
                    if last_err and last_err not in diag.actionable_message:
                        error_msg = f"Command `{' '.join(cmd)}` failed (exit {res.returncode}):\n{last_err}"
                    else:
                        error_msg = f"Command `{' '.join(cmd)}` failed: {diag.actionable_message}"
                    logger.error(error_msg)
                    raise EnvironmentSetupError(
                        message=error_msg,
                        diagnosis=diag,
                        command=cmd,
                        stdout=res.stdout,
                        stderr=res.stderr
                    )

        return results
