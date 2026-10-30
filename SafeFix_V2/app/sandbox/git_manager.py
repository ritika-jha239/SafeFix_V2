import asyncio
import base64
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import git
import httpx

logger = logging.getLogger("safefix.git")

def sanitize_secret(text: str) -> str:
    """Redacts secrets, GitHub tokens, and API keys from error messages, URLs, and logs."""
    if not text:
        return ""
    # Redact GitHub tokens (ghp_..., github_pat_..., x-access-token:...)
    s = re.sub(r"x-access-token:([^@]+)@", "x-access-token:***@", text)
    s = re.sub(r"(ghp_[a-zA-Z0-9_]{10,}|github_pat_[a-zA-Z0-9_]{20,})", "ghp_***REDACTED_GH_TOKEN***", s)
    s = re.sub(r"(nvapi-[a-zA-Z0-9_\-]{15,})", "nvapi_***REDACTED_NVIDIA_KEY***", s)
    s = re.sub(r"(gsk_[a-zA-Z0-9_\-]{15,})", "gsk_***REDACTED_GROQ_KEY***", s)
    s = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]{15,}", "Bearer ***REDACTED***", s)
    return s

class GitManager:
    @staticmethod
    def _auth_environment(token: Optional[str]) -> Dict[str, str]:
        """Pass GitHub authentication to a child Git process without saving it in .git/config."""
        env = os.environ.copy()
        if token:
            encoded = base64.b64encode(f"x-access-token:{token.strip()}".encode()).decode()
            env.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader", "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {encoded}", "GIT_TERMINAL_PROMPT": "0"})
        return env
    """Manages git clone, branching, diffs, commits, and GitHub Pull Request operations with strict safety guarantees."""

    def __init__(self, workspace_path: Path):
        self.workspace_path = Path(workspace_path)
        self.repo: Optional[git.Repo] = None

    @staticmethod
    def parse_issue_url(url: str) -> Optional[Dict[str, Any]]:
        """Extracts owner, repo, and issue_number from GitHub issue URL."""
        pattern = r"github\.com/([^/]+)/([^/]+)/issues/(\d+)"
        match = re.search(pattern, url.strip())
        if not match:
            return None
        return {
            "owner": match.group(1),
            "repo": match.group(2),
            "issue_number": int(match.group(3)),
            "clone_url": f"https://github.com/{match.group(1)}/{match.group(2)}.git"
        }

    @staticmethod
    async def fetch_issue_details(owner: str, repo: str, issue_number: int, token: Optional[str] = None) -> Dict[str, Any]:
        """Fetches issue title, description, labels from GitHub API."""
        endpoint = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SafeFix-PR-Agent"
        }
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(endpoint, headers=headers)
            if resp.status_code != 200:
                raise ValueError(f"GitHub API returned {resp.status_code}: {sanitize_secret(resp.text)}")
            return resp.json()

    @staticmethod
    async def get_default_branch(owner: str, repo: str, token: Optional[str] = None) -> str:
        """Determines the default branch (e.g. main or master) from GitHub API."""
        endpoint = f"https://api.github.com/repos/{owner}/{repo}"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SafeFix-PR-Agent"
        }
        if token and token.strip():
            headers["Authorization"] = f"Bearer {token.strip()}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(endpoint, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("default_branch", "main")
        except Exception:
            pass
        return "main"

    def init_local_repo(self, initial_commit_msg: str = "Initial commit with failing issue") -> git.Repo:
        """Initializes a local git repository in workspace for demo or offline test runs."""
        self.repo = git.Repo.init(str(self.workspace_path))
        self.repo.config_writer().set_value("user", "name", "SafeFix Agent").release()
        self.repo.config_writer().set_value("user", "email", "agent@safefix.dev").release()
        self.repo.git.add(A=True)
        self.repo.index.commit(initial_commit_msg)
        return self.repo

    async def clone_repo(self, clone_url: str, token: Optional[str] = None) -> git.Repo:
        """Clones a remote repository shallowly into the isolated sandbox."""
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path, ignore_errors=True)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        url = clone_url

        # Run git clone asynchronously
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--depth", "1", url, str(self.workspace_path),
            env=self._auth_environment(token),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            clean_err = sanitize_secret(stderr.decode("utf-8", errors="replace"))
            raise RuntimeError(f"Git clone failed: {clean_err}")

        self.repo = git.Repo(str(self.workspace_path))
        self.repo.config_writer().set_value("user", "name", "SafeFix Agent").release()
        self.repo.config_writer().set_value("user", "email", "agent@safefix.dev").release()
        return self.repo

    def get_diff(self) -> Dict[str, Any]:
        """Gets git diff summary, list of modified files, and raw unified diff."""
        if not self.repo:
            self.repo = git.Repo(str(self.workspace_path))

        diff_text = ""
        try:
            # Stage untracked files with intent-to-add so diff includes new files
            try:
                self.repo.git.add("-N", ".")
            except Exception:
                pass

            # 1. Check unstaged working tree diff
            diff_text = self.repo.git.diff()

            # 2. If empty, check staged / cached diff
            if not diff_text:
                diff_text = self.repo.git.diff("--cached")

            # 3. If still empty, check diff against previous commit HEAD~1
            if not diff_text:
                try:
                    commits = list(self.repo.iter_commits(max_count=2))
                    if len(commits) > 1:
                        diff_text = self.repo.git.diff("HEAD~1")
                except Exception:
                    diff_text = ""
        except Exception:
            diff_text = ""

        # Analyze diff
        files_changed = []
        additions = 0
        deletions = 0

        for line in diff_text.splitlines():
            if line.startswith("+++ b/"):
                files_changed.append(line[6:].strip())
            elif line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
                files_changed.append(line[4:].lstrip("b/").strip())
            elif line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

        return {
            "diff_text": diff_text,
            "files_changed": sorted(list(set(files_changed))),
            "additions": additions,
            "deletions": deletions
        }

    def create_branch(self, branch_name: str):
        """Creates and checks out a fresh uniquely named branch without force."""
        if not self.repo:
            self.repo = git.Repo(str(self.workspace_path))
        # Create fresh branch
        if branch_name not in [b.name for b in self.repo.branches]:
            new_branch = self.repo.create_head(branch_name)
            new_branch.checkout()
        else:
            self.repo.git.checkout(branch_name)

    def commit_changes(self, commit_message: str):
        if not self.repo:
            self.repo = git.Repo(str(self.workspace_path))
        self.repo.git.add(A=True)
        if self.repo.is_dirty():
            return self.repo.index.commit(commit_message)
        return None

    async def push_and_create_pr(
        self,
        owner: str,
        repo_name: str,
        branch_name: str,
        base_branch: str,
        pr_title: str,
        pr_body: str,
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Pushes branch to GitHub without force-pushing and opens a Pull Request.
        
        Handles forks when the original repository is not directly writable.
        """
        if not token or not token.strip():
            # Simulated PR creation for demo / without token
            return {
                "success": True,
                "simulated": True,
                "pr_url": f"https://github.com/{owner}/{repo_name}/pull/new/{branch_name}",
                "pr_number": 99,
                "branch": branch_name,
                "message": "Simulated PR created successfully! (Provide a GitHub Token in settings to push live to GitHub)."
            }

        if not self.repo:
            self.repo = git.Repo(str(self.workspace_path))

        headers = {
            "Authorization": f"Bearer {token.strip()}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SafeFix-PR-Agent"
        }

        # 1. Determine user identity
        user_login = owner
        async with httpx.AsyncClient(timeout=15.0) as client:
            user_resp = await client.get("https://api.github.com/user", headers=headers)
            if user_resp.status_code == 200:
                user_login = user_resp.json().get("login", owner)

        # 2. Attempt push to origin
        remote_url = f"https://github.com/{owner}/{repo_name}.git"
        if "origin" in [r.name for r in self.repo.remotes]:
            origin = self.repo.remote("origin")
            origin.set_url(remote_url)
        else:
            origin = self.repo.create_remote("origin", remote_url)

        push_success = False
        target_head = branch_name

        try:
            # Never force-push; use safe standard push
            self.repo.git.push("-u", "origin", branch_name, env=self._auth_environment(token))
            push_success = True
        except Exception as e:
            logger.info(f"Direct push to {owner}/{repo_name} failed. Attempting fork workflow: {sanitize_secret(str(e))}")

        # 3. Fork fallback if direct push was denied
        if not push_success:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    fork_resp = await client.post(f"https://api.github.com/repos/{owner}/{repo_name}/forks", headers=headers)
                    if fork_resp.status_code not in [200, 202]:
                        return {
                            "success": False,
                            "simulated": False,
                            "error": f"Failed to fork repository: {sanitize_secret(fork_resp.text)}"
                        }
                    fork_data = fork_resp.json()
                    fork_owner = fork_data.get("owner", {}).get("login", user_login)
                    fork_remote_url = f"https://github.com/{fork_owner}/{repo_name}.git"

                    if "fork" in [r.name for r in self.repo.remotes]:
                        fork_remote = self.repo.remote("fork")
                        fork_remote.set_url(fork_remote_url)
                    else:
                        fork_remote = self.repo.create_remote("fork", fork_remote_url)

                    # Push to fork
                    self.repo.git.push("-u", "fork", branch_name, env=self._auth_environment(token))
                    target_head = f"{fork_owner}:{branch_name}"
                    push_success = True
            except Exception as e:
                return {
                    "success": False,
                    "simulated": False,
                    "error": f"Failed to push branch to repository or fork: {sanitize_secret(str(e))}"
                }

        # 4. Open Pull Request on GitHub
        pr_endpoint = f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
        payload = {
            "title": pr_title,
            "body": pr_body,
            "head": target_head,
            "base": base_branch
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(pr_endpoint, headers=headers, json=payload)
            if resp.status_code in [200, 201]:
                data = resp.json()
                return {
                    "success": True,
                    "simulated": False,
                    "pr_url": data.get("html_url"),
                    "pr_number": data.get("number"),
                    "branch": branch_name,
                    "message": f"Successfully created Pull Request #{data.get('number')} on GitHub!"
                }
            else:
                return {
                    "success": False,
                    "simulated": False,
                    "error": f"GitHub API error ({resp.status_code}): {sanitize_secret(resp.text)}"
                }
