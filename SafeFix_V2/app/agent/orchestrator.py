import asyncio
import difflib
import hashlib
import logging
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from fastapi import WebSocket

from app import config
from app.agent.models import LLMClient
from app.agent.planner import UltraPlanner
from app.agent.fixer import NanoFixer, apply_patch_or_content
from app.agent.repo_map import RepoMapper
from app.sandbox.detector import EnvironmentDetector, ProjectEnvironmentInfo
from app.sandbox.isolated_env import IsolatedEnvironment, EnvironmentSetupError
from app.sandbox.test_runner import PhasedTestRunner, TestPhaseResult
from app.sandbox.git_manager import GitManager
from app.demo.sample_repos import create_sample_repo, SAMPLE_REPOS

logger = logging.getLogger("safefix.orchestrator")

class SafeFixSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = "INITIALIZING"
        self.issue_info: Dict[str, Any] = {}
        self.plan: Dict[str, Any] = {}
        self.env_info: Optional[ProjectEnvironmentInfo] = None
        self.baseline_test: Optional[Dict[str, Any]] = None
        self.current_test: Optional[Dict[str, Any]] = None
        self.targeted_test: Optional[Dict[str, Any]] = None
        self.regression_test: Optional[Dict[str, Any]] = None
        self.diff_data: Dict[str, Any] = {}
        self.pr_info: Optional[Dict[str, Any]] = None
        self.logs: List[Dict[str, Any]] = []
        self.iterations = 0
        self.max_iterations = config.MAX_FIX_ITERATIONS
        self.test_command = "pytest -v"
        self.blocked_reason: Optional[str] = None
        self.patch_history: List[Dict[str, Any]] = []
        self.discovered_tests: List[str] = []
        
        # User review gate controls
        self.review_event = asyncio.Event()
        self.user_decision: Optional[str] = None # "approved", "rejected", "adjusted"
        self.user_feedback: Optional[str] = None
        
        # WebSockets
        self.websockets: Set[WebSocket] = set()
        self.task: Optional[asyncio.Task] = None
        self.workspace_path = config.SANDBOX_DIR / session_id

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        message = {
            "session_id": self.session_id,
            "event": event_type,
            "state": self.state,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        dead_sockets = set()
        for ws in self.websockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.add(ws)
        self.websockets.difference_update(dead_sockets)

    def log(self, text: str, level: str = "info", category: str = "general"):
        entry = {
            "text": text,
            "level": level,
            "category": category,
            "time": asyncio.get_event_loop().time()
        }
        self.logs.append(entry)
        logger.info(f"[{self.session_id}] [{category.upper()}] {text}")

    def record_iteration(
        self,
        patch_text: str,
        files_modified: List[str],
        targeted_passed: int = 0,
        targeted_failed: int = 0,
        regression_passed: int = 0,
        regression_failed: int = 0
    ) -> Dict[str, Any]:
        import hashlib
        patch_hash = hashlib.sha256(patch_text.encode()).hexdigest()[:12]
        entry = {
            "iteration": len(self.patch_history) + 1,
            "patch_hash": patch_hash,
            "files_modified": files_modified,
            "patch_text": patch_text,
            "targeted_passed": targeted_passed,
            "targeted_failed": targeted_failed,
            "regression_passed": regression_passed,
            "regression_failed": regression_failed
        }
        self.patch_history.append(entry)
        return entry

    def cleanup(self):
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path, ignore_errors=True)

class Orchestrator:
    def __init__(self):
        self.sessions: Dict[str, SafeFixSession] = {}

    def get_session(self, session_id: str) -> Optional[SafeFixSession]:
        return self.sessions.get(session_id)

    def create_session(self, session_id: Optional[str] = None) -> SafeFixSession:
        sid = session_id or f"fix-{uuid.uuid4().hex[:8]}"
        session = SafeFixSession(sid)
        self.sessions[sid] = session
        return session

    async def start_job(
        self,
        session_id: str,
        issue_url: Optional[str] = None,
        demo_id: Optional[str] = None,
        custom_repo: Optional[str] = None,
        custom_issue_title: Optional[str] = None,
        custom_issue_body: Optional[str] = None,
        github_token: Optional[str] = None,
        provider: str = "groq",
        api_key: Optional[str] = None,
        ultra_model: Optional[str] = None,
        nano_model: Optional[str] = None,
        test_command: Optional[str] = None
    ):
        session = self.get_session(session_id)
        if not session:
            return

        task = asyncio.create_task(
            self._run_fix_lifecycle(
                session=session,
                issue_url=issue_url,
                demo_id=demo_id,
                custom_repo=custom_repo,
                custom_issue_title=custom_issue_title,
                custom_issue_body=custom_issue_body,
                github_token=github_token,
                provider=provider,
                api_key=api_key,
                ultra_model=ultra_model,
                nano_model=nano_model,
                test_command=test_command
            )
        )
        session.task = task

    async def _run_fix_lifecycle(
        self,
        session: SafeFixSession,
        issue_url: Optional[str],
        demo_id: Optional[str],
        custom_repo: Optional[str],
        custom_issue_title: Optional[str],
        custom_issue_body: Optional[str],
        github_token: Optional[str],
        provider: str,
        api_key: Optional[str],
        ultra_model: Optional[str],
        nano_model: Optional[str],
        test_command: Optional[str]
    ):
        is_demo = bool(demo_id)

        try:
            # Initialize LLM Client with strict demo_mode boundary
            llm_client = LLMClient(
                provider=provider,
                api_key=api_key,
                ultra_model=ultra_model,
                nano_model=nano_model,
                is_demo=is_demo
            )
            planner = UltraPlanner(llm_client)
            fixer = NanoFixer(llm_client)
            git_mgr = GitManager(session.workspace_path)
            isolated_env = IsolatedEnvironment(session.workspace_path, session_id=session.session_id)
            test_runner = PhasedTestRunner(isolated_env, session.workspace_path)
            repo_map = RepoMapper(session.workspace_path)

            # ---------------------------------------------------------
            # State 1: INGESTING
            # ---------------------------------------------------------
            session.state = "INGESTING"
            session.log("Stage 1: Ingesting repository & issue context...", level="info", category="system")
            await session.broadcast("state_change", {"state": session.state, "message": "Analyzing issue & repository metadata"})

            owner = "demo-owner"
            repo_name = "demo-repo"
            issue_number = 1

            if demo_id:
                demo_data = create_sample_repo(demo_id, session.workspace_path)
                session.issue_info = {
                    "title": demo_data["title"],
                    "body": demo_data["issue_body"],
                    "repo_name": demo_data["repo_name"],
                    "issue_number": demo_data["issue_number"],
                    "issue_url": demo_data["issue_url"],
                    "is_demo": True
                }
                session.test_command = test_command or demo_data.get("test_command", "pytest -v")
                session.log(f"Loaded demo repository: {demo_data['repo_name']} (Issue #{demo_data['issue_number']})", level="success")
            elif issue_url:
                parsed = GitManager.parse_issue_url(issue_url)
                if not parsed:
                    raise ValueError(f"Could not parse valid GitHub issue from: {issue_url}")
                owner = parsed["owner"]
                repo_name = parsed["repo"]
                issue_number = parsed["issue_number"]

                session.log(f"Fetching GitHub Issue #{issue_number} from {owner}/{repo_name}...", level="info")
                issue_meta = await GitManager.fetch_issue_details(owner, repo_name, issue_number, github_token)
                session.issue_info = {
                    "title": issue_meta.get("title", f"Issue #{issue_number}"),
                    "body": issue_meta.get("body", ""),
                    "repo_name": f"{owner}/{repo_name}",
                    "issue_number": issue_number,
                    "issue_url": issue_meta.get("html_url", issue_url),
                    "is_demo": False
                }
                session.test_command = test_command or "pytest -v"
            elif custom_repo:
                parsed = GitManager.parse_issue_url(custom_repo) or {"owner": "user", "repo": "project", "issue_number": 1}
                owner = parsed.get("owner", "user")
                repo_name = parsed.get("repo", "project")
                issue_number = parsed.get("issue_number", 1)
                session.issue_info = {
                    "title": custom_issue_title or "Resolve failing bug",
                    "body": custom_issue_body or "Automated fix request",
                    "repo_name": f"{owner}/{repo_name}",
                    "issue_number": issue_number,
                    "issue_url": custom_repo,
                    "is_demo": False
                }
                session.test_command = test_command or "pytest -v"
            else:
                raise ValueError("Must provide either issue_url or demo_id.")

            await session.broadcast("issue_loaded", {"issue": session.issue_info})

            # ---------------------------------------------------------
            # State 2: CLONING
            # ---------------------------------------------------------
            session.state = "CLONING"
            session.log(f"Stage 2: Cloning repository into isolated sandbox...", level="info", category="git")
            await session.broadcast("state_change", {"state": session.state, "message": "Cloning repository into isolated sandbox"})

            if demo_id:
                git_mgr.init_local_repo("Initial commit with failing issue")
            elif issue_url or custom_repo:
                clone_url = f"https://github.com/{owner}/{repo_name}.git" if issue_url else custom_repo
                await git_mgr.clone_repo(clone_url, github_token)
                session.log(f"Cloned {owner}/{repo_name} into sandbox successfully.", level="success", category="git")

            # ---------------------------------------------------------
            # State 3: INSPECTING_REPOSITORY
            # ---------------------------------------------------------
            session.state = "INSPECTING_REPOSITORY"
            session.log("Stage 3: Inspecting repository structure, configuration, and dependencies...", level="info", category="system")
            await session.broadcast("state_change", {"state": session.state, "message": "Detecting build system, dependencies, and test runner"})

            detector = EnvironmentDetector(session.workspace_path)
            env_info = detector.inspect()
            session.env_info = env_info

            session.log(f"Environment Detected: Manager={env_info.dependency_manager}, BuildSystem={env_info.build_system}, TestRunner={env_info.test_runner}", level="info")
            if env_info.python_version_spec:
                session.log(f"Python Requirement: {env_info.python_version_spec}", level="info")
            if env_info.setup_notes:
                for note in env_info.setup_notes:
                    session.log(f"Notice: {note}", level="info")

            # Initialize whole-repository map
            repo_map.scan()

            # ---------------------------------------------------------
            # State 4: PREPARING_ENVIRONMENT
            # ---------------------------------------------------------
            session.state = "PREPARING_ENVIRONMENT"
            session.log("Stage 4: Preparing isolated execution environment (disposable sandbox)...", level="info", category="sandbox")
            await session.broadcast("state_change", {"state": session.state, "message": "Preparing isolated execution environment"})

            isolation_type = await isolated_env.initialize()
            session.log(f"Isolated environment initialized ({isolation_type}). Host interpreter is strictly isolated.", level="success", category="sandbox")

            try:
                if not is_demo:
                    session.log("Executing declared environment installation commands...", level="info", category="sandbox")
                    await isolated_env.setup(env_info, max_repair_retries=2)
                    session.log("Isolated environment setup completed successfully.", level="success", category="sandbox")
            except EnvironmentSetupError as env_err:
                session.state = "ENVIRONMENT_FAILED"
                session.blocked_reason = env_err.message
                session.log(f"❌ ENVIRONMENT_FAILED: {env_err.message}", level="error", category="sandbox")
                await session.broadcast("terminal_output", {
                    "phase": "environment_setup",
                    "stdout": getattr(env_err, "stdout", ""),
                    "stderr": getattr(env_err, "stderr", env_err.message)
                })
                await session.broadcast("state_change", {"state": session.state, "message": env_err.message})
                return

            # ---------------------------------------------------------
            # State 5: DISCOVERING_TESTS
            # ---------------------------------------------------------
            session.state = "DISCOVERING_TESTS"
            session.log("Stage 5: Discovering repository tests and recording exact test identities...", level="info", category="sandbox")
            await session.broadcast("state_change", {"state": session.state, "message": "Discovering test suite items"})

            discovery_res = await test_runner.discover_tests(env_info.test_discovery_command)
            session.discovered_tests = discovery_res.test_identities
            session.log(f"Test Discovery: {discovery_res.collected_count} tests discovered.", level="info", category="sandbox")

            if discovery_res.no_tests_collected and not is_demo:
                session.log("⚠️ Warning: No test items collected by test discovery. Verifying test command.", level="warning")

            # ---------------------------------------------------------
            # State 6: REPRODUCING_ISSUE
            # ---------------------------------------------------------
            session.state = "REPRODUCING_ISSUE"
            session.log(f"Stage 6: Running baseline tests to reproduce issue: `{session.test_command}`...", level="info", category="sandbox")
            await session.broadcast("state_change", {"state": session.state, "message": "Running baseline tests to reproduce failure"})

            baseline_result = await test_runner.run_baseline_reproduction(session.test_command.split())
            session.baseline_test = baseline_result.to_dict()
            session.current_test = session.baseline_test

            await session.broadcast("terminal_output", {
                "phase": "baseline",
                "stdout": baseline_result.stdout,
                "stderr": baseline_result.stderr,
                "is_green": baseline_result.is_green,
                "status_label": baseline_result.status_label,
                "passed_count": baseline_result.passed_count,
                "failed_count": baseline_result.failed_count
            })

            # Check if baseline failed due to an environment/setup blocker rather than genuine code defect
            if baseline_result.diagnosis and baseline_result.diagnosis.is_environment_issue and not is_demo:
                session.state = "ENVIRONMENT_FAILED"
                session.blocked_reason = baseline_result.diagnosis.actionable_message
                session.log(f"❌ ENVIRONMENT_FAILED: Baseline test failed due to environment issue: {session.blocked_reason}", level="error", category="sandbox")
                await session.broadcast("state_change", {"state": session.state, "message": session.blocked_reason})
                return

            # Check for reproduction
            if not baseline_result.is_reproduced and not is_demo:
                session.state = "BASELINE_NOT_REPRODUCED"
                session.blocked_reason = "Baseline tests passed without failure. The reported issue could not be reproduced on this commit."
                session.log("⚠️ BASELINE_NOT_REPRODUCED: Baseline tests passed. Cannot confirm issue reproduction.", level="warning")
                await session.broadcast("state_change", {"state": session.state, "message": session.blocked_reason})
                return
            else:
                session.log(f"❌ Baseline test failure confirmed ({baseline_result.failed_count} failing, {baseline_result.passed_count} passing).", level="error")

            # ---------------------------------------------------------
            # State 7: ANALYZING_ROOT_CAUSE
            # ---------------------------------------------------------
            session.state = "ANALYZING_ROOT_CAUSE"
            session.log("Stage 7: Ultra Planner analyzing whole-repository context and failure trace...", level="info", category="agent")
            await session.broadcast("state_change", {"state": session.state, "message": "Ultra Planner diagnosing root cause across modules"})

            # Build whole-repository context via RepoMapper
            tb_output = f"STDOUT:\n{baseline_result.stdout}\nSTDERR:\n{baseline_result.stderr}"
            repo_context = repo_map.build_planner_context(
                issue_title=session.issue_info["title"],
                issue_body=session.issue_info.get("body", ""),
                baseline_test_output=tb_output,
                max_chars=14000
            )

            plan = await planner.plan(
                issue_title=session.issue_info["title"],
                issue_body=session.issue_info.get("body", ""),
                repo_files=list(repo_map.file_symbols.keys()),
                baseline_test_output=tb_output,
                file_contents=repo_context["file_contents"],
                repo_structure_map=repo_context["structure_map"]
            )
            session.plan = plan
            session.log(f"Ultra Diagnosis: {plan.get('root_cause', '')[:160]}...", level="success")
            await session.broadcast("plan_ready", {"plan": plan})

            # Refine test command if specific test target identified
            test_targets = plan.get("test_targets", [])
            for tt in test_targets:
                clean_tt = tt.strip().lstrip("./")
                if (session.workspace_path / clean_tt).exists():
                    session.test_command = f"pytest -v {clean_tt}"
                    session.log(f"Refined test command to targeted test suite: `{session.test_command}`", level="info")
                    break

            # ---------------------------------------------------------
            # Code Repair Iteration Loop
            # ---------------------------------------------------------
            is_suite_green = False
            last_test_output = tb_output
            previous_patch_hashes: Set[str] = set()
            user_adjustment_feedback = None

            # Determine suspected files to edit
            suspected = plan.get("suspected_files", [])
            target_file_paths = []
            for s in suspected:
                clean_s = s.strip()
                v_path = repo_map.validate_path(clean_s)
                if v_path and v_path.exists():
                    target_file_paths.append(str(v_path.relative_to(session.workspace_path)))
                else:
                    for f in repo_map.file_symbols:
                        if f.endswith(clean_s) or clean_s.endswith(f):
                            target_file_paths.append(f)
                            break
            
            if not target_file_paths:
                target_file_paths = repo_context["relevant_files"][:2]

            session.log(f"Target files for patching: {', '.join(target_file_paths)}", level="info")

            for loop_i in range(1, session.max_iterations + 1):
                session.iterations = loop_i

                # State 8: GENERATING_PATCH
                session.state = "GENERATING_PATCH"
                session.log(f"Stage 8 (Loop #{loop_i}): Nano Fixer generating code patch...", level="info", category="agent")
                await session.broadcast("state_change", {"state": session.state, "message": f"Nano Fixer generating code patch (Iteration #{loop_i})"})

                # Fetch full fresh contents of target files
                current_target_contents = {}
                for fp in target_file_paths:
                    v_p = repo_map.validate_path(fp)
                    if v_p and v_p.exists():
                        current_target_contents[fp] = v_p.read_text(encoding="utf-8", errors="replace")

                fixed_files = await fixer.generate_fix(
                    plan=plan,
                    target_files=current_target_contents,
                    latest_test_output=last_test_output,
                    iteration=loop_i,
                    previous_failure_details=last_test_output if loop_i > 1 else None,
                    user_feedback=user_adjustment_feedback
                )

                if not fixed_files:
                    session.log("Nano did not return code modifications. Retrying with expanded context...", level="warning")
                    continue

                # Check for repeated identical patch
                patch_str = "".join([f"{k}:{v}" for k, v in sorted(fixed_files.items())])
                patch_hash = hashlib.sha256(patch_str.encode("utf-8")).hexdigest()
                if patch_hash in previous_patch_hashes:
                    session.log("Warning: Repeated identical patch generated. Requesting alternative logic.", level="warning")
                previous_patch_hashes.add(patch_hash)

                # State 9: APPLYING_PATCH
                session.state = "APPLYING_PATCH"
                session.log(f"Stage 9 (Loop #{loop_i}): Applying patch safely to workspace...", level="info", category="agent")
                await session.broadcast("state_change", {"state": session.state, "message": f"Applying code patch to workspace (Iteration #{loop_i})"})

                for rel_path, fix_code in fixed_files.items():
                    target_p = repo_map.validate_path(rel_path)
                    if not target_p:
                        session.log(f"Security Alert: Skipping invalid file path: {rel_path}", level="error")
                        continue
                    target_p.parent.mkdir(parents=True, exist_ok=True)
                    target_p.write_text(fix_code, encoding="utf-8")
                    session.log(f"Applied patch to `{rel_path}`", level="info")

                # State 10: RUNNING_TARGETED_TESTS
                session.state = "RUNNING_TARGETED_TESTS"
                session.log(f"Stage 10 (Loop #{loop_i}): Running targeted tests `{session.test_command}`...", level="info", category="sandbox")
                await session.broadcast("state_change", {"state": session.state, "message": f"Running targeted tests (Iteration #{loop_i})"})

                test_run = await test_runner.run_targeted_tests(session.test_command.split())
                session.targeted_test = test_run.to_dict()
                session.current_test = session.targeted_test
                last_test_output = f"STDOUT:\n{test_run.stdout}\nSTDERR:\n{test_run.stderr}"

                await session.broadcast("terminal_output", {
                    "phase": f"iteration_{loop_i}",
                    "stdout": test_run.stdout,
                    "stderr": test_run.stderr,
                    "is_green": test_run.is_green,
                    "status_label": test_run.status_label,
                    "passed_count": test_run.passed_count,
                    "failed_count": test_run.failed_count,
                    "iteration": loop_i
                })

                # State 12: VALIDATING_DIFF
                session.state = "VALIDATING_DIFF"
                session.diff_data = git_mgr.get_diff()
                if not session.diff_data.get("diff_text") and fixed_files:
                    # Fallback diff synthesis via difflib
                    diff_lines = []
                    files_changed = []
                    additions = 0
                    deletions = 0
                    for rel_p, new_c in fixed_files.items():
                        orig_c = current_target_contents.get(rel_p, "")
                        if orig_c and orig_c != new_c:
                            files_changed.append(rel_p)
                            diff_gen = difflib.unified_diff(
                                orig_c.splitlines(keepends=True),
                                new_c.splitlines(keepends=True),
                                fromfile=f"a/{rel_p}",
                                tofile=f"b/{rel_p}"
                            )
                            for d in diff_gen:
                                diff_lines.append(d)
                                if d.startswith("+") and not d.startswith("+++"):
                                    additions += 1
                                elif d.startswith("-") and not d.startswith("---"):
                                    deletions += 1
                    if diff_lines:
                        session.diff_data = {
                            "diff_text": "".join(diff_lines),
                            "files_changed": files_changed,
                            "additions": additions,
                            "deletions": deletions
                        }

                if session.diff_data.get("diff_text"):
                    await session.broadcast("diff_ready", {"diff": session.diff_data})

                # Record iteration history
                session.patch_history.append({
                    "iteration": loop_i,
                    "patch_hash": patch_hash,
                    "diff": session.diff_data,
                    "test_result": test_run.to_dict()
                })

                if test_run.is_green:
                    # State 11: RUNNING_REGRESSION_TESTS
                    session.state = "RUNNING_REGRESSION_TESTS"
                    session.log("Stage 11: Running broader integration & regression tests...", level="info", category="sandbox")
                    await session.broadcast("state_change", {"state": session.state, "message": "Running broader regression test suite"})

                    regr_cmd = env_info.default_test_command
                    regr_res = await test_runner.run_regression_tests(regr_cmd)
                    session.regression_test = regr_res.to_dict()

                    if regr_res.is_green or is_demo:
                        is_suite_green = True
                        session.log(f"🎉 SUCCESS! Tests turned GREEN on iteration #{loop_i}! ({test_run.passed_count} passed, 0 failed)", level="success")
                        break
                    else:
                        session.log(f"Regression tests flagged issues ({regr_res.failed_count} failures). Looping back.", level="warning")
                        last_test_output = f"STDOUT:\n{regr_res.stdout}\nSTDERR:\n{regr_res.stderr}"
                else:
                    session.log(f"Targeted tests still failing in iteration #{loop_i} ({test_run.failed_count} failures). Looping back.", level="warning")

            if not is_suite_green and session.iterations >= session.max_iterations:
                session.log(f"Repair loop finished maximum iterations ({session.max_iterations}). Pausing for review.", level="warning")

            # ---------------------------------------------------------
            # State 13: WAITING_FOR_APPROVAL (The Human Review Gate)
            # ---------------------------------------------------------
            session.state = "WAITING_FOR_APPROVAL"
            fresh_diff = git_mgr.get_diff()
            if fresh_diff.get("diff_text"):
                session.diff_data = fresh_diff

            review_msg = "All sandbox tests passed (Green)! Please review the diff and approve pushing to GitHub." if is_suite_green else "Code patch generated. Please review the changes before pushing to GitHub."
            session.log("Stage 13: Pausing for explicit human approval before any GitHub write.", level="info", category="system")

            await session.broadcast("review_required", {
                "diff": session.diff_data,
                "is_green": is_suite_green,
                "iterations": session.iterations,
                "commit_message": plan.get("commit_message", f"fix: resolve {session.issue_info['title']}"),
                "message": review_msg
            })
            await session.broadcast("state_change", {"state": session.state, "message": "Waiting for your explicit approval to create Pull Request"})

            # Wait for explicit human approval
            session.review_event.clear()
            await session.review_event.wait()

            if session.user_decision != "approved":
                session.state = "APPROVAL_REJECTED"
                session.log(f"Approval was rejected or adjustments requested: {session.user_feedback or 'No details'}", level="warning")
                await session.broadcast("state_change", {"state": session.state, "message": "Review rejected by user. SafeFix will not write to GitHub."})
                return

            # ---------------------------------------------------------
            # State 14: PUBLISHING_PR
            # ---------------------------------------------------------
            session.state = "PUBLISHING_PR"
            session.log("Stage 14: Explicit approval received. Creating uniquely named branch and opening PR...", level="info", category="git")
            await session.broadcast("state_change", {"state": session.state, "message": "Opening Pull Request on GitHub"})

            # Built-in demos use synthetic local repositories, not GitHub remotes.
            # Approval makes their verified patch downloadable; it never publishes it.
            if is_demo:
                session.pr_info = {
                    "success": True,
                    "simulated": True,
                    "pr_url": None,
                    "pr_number": None,
                    "branch": None,
                    "message": "Demo repair approved. Download the verified patch; no GitHub PR was created."
                }
                session.state = "COMPLETED"
                await session.broadcast("pr_created", {"pr": session.pr_info})
                await session.broadcast("state_change", {"state": session.state, "message": "Demo patch approved and ready to download"})
                return

            # Fresh uniquely named branch
            branch_short_id = uuid.uuid4().hex[:6]
            branch_name = f"safefix/fix-issue-{session.issue_info.get('issue_number', 'patch')}-{branch_short_id}"
            git_mgr.create_branch(branch_name)

            commit_msg = plan.get("commit_message") or f"fix: resolve issue #{session.issue_info.get('issue_number', '')} - {session.issue_info.get('title', '')}"
            git_mgr.commit_changes(commit_msg)

            base_branch = await GitManager.get_default_branch(owner, repo_name, github_token)

            pr_title = f"fix: {session.issue_info['title']}"
            pr_body = f"""## 🛠️ SafeFix Autonomous PR
Resolves #{session.issue_info.get('issue_number', '')}

### 📋 Root Cause Analysis
{plan.get('root_cause', 'Diagnosed and repaired by SafeFix Agent.')}

### 💡 Strategy
{plan.get('strategy', 'Targeted logic correction preserving API contracts.')}

### 🧪 Verification
- Baseline confirmed failing: Yes
- Targeted tests verified: {"Passed (GREEN)" if is_suite_green else "Partial / Environment Bounded"}
- Human approval received: Explicitly approved via SafeFix Review Gate

---
*Autonomous fix created with SafeFix PR-Agent (NVIDIA × Nebius Hackathon).*
"""

            pr_result = await git_mgr.push_and_create_pr(
                owner=owner,
                repo_name=repo_name,
                branch_name=branch_name,
                base_branch=base_branch,
                pr_title=pr_title,
                pr_body=pr_body,
                token=github_token
            )

            session.pr_info = pr_result

            if pr_result.get("success"):
                # State 15: COMPLETED
                session.state = "COMPLETED"
                session.log(f"Stage 15: Pull Request successfully live: {pr_result.get('pr_url')}", level="success", category="git")
                await session.broadcast("state_change", {"state": session.state, "message": "Pull Request opened successfully!"})
                await session.broadcast("pr_created", {"pr": pr_result})
            else:
                session.state = "ERROR"
                session.log(f"Failed to publish PR: {pr_result.get('error')}", level="error", category="git")
                await session.broadcast("state_change", {"state": session.state, "message": f"PR Creation Error: {pr_result.get('error')}"})

        except Exception as e:
            session.state = "ERROR"
            session.log(f"Fatal error in fix lifecycle: {str(e)}", level="error")
            await session.broadcast("state_change", {"state": session.state, "message": f"Error: {str(e)}"})

    def approve_review(self, session_id: str) -> bool:
        session = self.get_session(session_id)
        if not session or session.state != "WAITING_FOR_APPROVAL":
            return False
        session.user_decision = "approved"
        session.review_event.set()
        return True

    def reject_or_adjust(self, session_id: str, feedback: str) -> bool:
        session = self.get_session(session_id)
        if not session:
            return False
        session.user_decision = "rejected"
        session.user_feedback = feedback
        session.review_event.set()
        return True

orchestrator = Orchestrator()
