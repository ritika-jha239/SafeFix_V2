import json
import logging
from typing import Dict, Any, List, Optional
from app.agent.models import LLMClient

logger = logging.getLogger("safefix.planner")

ULTRA_SYSTEM_PROMPT = """You are SafeFix Ultra, an elite Principal Software Engineer & Architecture Reasoning Agent specializing in autonomous Python debugging.
Your role is to deeply analyze GitHub issues, inspect the repository structure, examine failing test traces, and produce an ironclad Root Cause Analysis & Fix Plan.

You must output a structured JSON response with the following format:
{
  "issue_summary": "Concise summary of the problem reported in the issue",
  "root_cause": "Exact explanation of why the code is failing or bug is occurring",
  "suspected_files": ["path/to/buggy_file.py"],
  "strategy": "Precise algorithm/logic adjustments required to fix the issue without breaking existing tests",
  "test_targets": ["tests/test_file.py"],
  "commit_message": "fix: descriptive conventional commit message"
}

Be thorough, precise, and preserve existing API contracts.
"""

class UltraPlanner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def plan(
        self,
        issue_title: str,
        issue_body: str,
        repo_files: List[str],
        baseline_test_output: str,
        file_contents: Dict[str, str],
        repo_structure_map: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the Ultra reasoning model to diagnose the root cause and prepare a fix plan."""
        
        # Build structural overview
        if repo_structure_map:
            files_overview = repo_structure_map
        else:
            files_overview = "\n".join([f"- {f}" for f in repo_files[:25]])
            if len(repo_files) > 25:
                files_overview += f"\n- ... and {len(repo_files) - 25} more files"
        
        code_snippets = []
        for path, code in list(file_contents.items())[:5]:
            # Keep snippets concise to fit safely within LLM rate limits
            trimmed = code[:2500] if len(code) > 2500 else code
            code_snippets.append(f"### File: {path}\n```python\n{trimmed}\n```")
        
        snippets_text = "\n\n".join(code_snippets)

        prompt = f"""### Target GitHub Issue:
Title: {issue_title}

Description:
{issue_body[:1800]}

### Repository Structural & Symbol Map:
{files_overview}

### Baseline Test Results / Traceback:
```
{baseline_test_output[:1800]}
```

### Relevant Code Definitions:
{snippets_text}

Analyze the problem across modules and return the JSON root cause diagnosis and fix plan.
"""

        messages = [
            {"role": "system", "content": ULTRA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        raw_response = await self.llm.call(
            messages=messages,
            role="ultra",
            temperature=0.1
        )

        # Parse JSON from response
        try:
            clean_json = raw_response.strip()
            if "```json" in clean_json:
                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_json:
                clean_json = clean_json.split("```")[1].split("```")[0].strip()
            
            # Find json bracket boundaries
            start_idx = clean_json.find("{")
            end_idx = clean_json.rfind("}")
            if start_idx != -1 and end_idx != -1:
                clean_json = clean_json[start_idx:end_idx+1]

            plan_data = json.loads(clean_json)
            return plan_data
        except Exception as e:
            logger.warning(f"Could not parse Ultra JSON directly ({e}). Using extracted plan.")
            
            # Heuristic regex extraction fallback
            summary_m = re.search(r'"issue_summary":\s*"([^"]+)"', raw_response)
            root_m = re.search(r'"root_cause":\s*"([^"]+)"', raw_response)
            strat_m = re.search(r'"strategy":\s*"([^"]+)"', raw_response)
            suspected_m = re.findall(r'"([a-zA-Z0-9_\-\./\\]+\.py)"', raw_response)
            
            suspected_candidates = [f for f in suspected_m if any(f.endswith(rf) or rf.endswith(f) for rf in repo_files)]
            
            return {
                "issue_summary": summary_m.group(1) if summary_m else issue_title,
                "root_cause": root_m.group(1) if root_m else "Issue diagnosed from traceback and code analysis.",
                "suspected_files": list(set(suspected_candidates)) if suspected_candidates else list(file_contents.keys())[:2],
                "strategy": strat_m.group(1) if strat_m else "Apply minimal targeted fix to resolve failing assertion.",
                "test_targets": [f for f in repo_files if "test" in f.lower()][:1],
                "commit_message": f"fix: resolve {issue_title[:50]}"
            }
