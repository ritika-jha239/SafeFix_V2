import re
import logging
from pathlib import Path
from typing import Dict, Any, List
from app.agent.models import LLMClient

logger = logging.getLogger("safefix.fixer")

NANO_SYSTEM_PROMPT = """You are SafeFix Nano, an ultra-fast code repair and test-fix execution agent.
Your mission is to generate the exact minimal code repair to fix the failing tests.

CRITICAL INSTRUCTIONS FOR RATE-LIMIT EFFICIENCY:
1. Output ONLY the minimal SEARCH/REPLACE block needed to fix the bug.
2. DO NOT output unchanged code, unchanged imports, or entire files unless the file is under 40 lines.
3. Keep SEARCH blocks small (5-20 lines) to stay strictly within token limits.

Format your fix inside a fenced code block with the target file path:
```python:path/to/file.py
<<<<<<< SEARCH
# exact lines in the existing code to replace
=======
# corrected replacement code
>>>>>>> REPLACE
```
"""

def apply_patch_or_content(original_content: str, fix_block: str) -> str:
    """Applies either search/replace blocks, targeted function replacement, or full file replacement."""
    if not fix_block or not fix_block.strip():
        return original_content

    original_norm = original_content.replace("\r\n", "\n")
    fix_norm = fix_block.replace("\r\n", "\n")

    # 1. Search & Replace blocks (with optional trailing REPLACE fence in case of truncation)
    if "<<<<<<< SEARCH" in fix_norm:
        pattern = r"<<<<<<< SEARCH[ \t]*\n(.*?)\n=======[ \t]*\n(.*?)(?:\n>>>>>>> REPLACE|\Z)"
        matches = re.findall(pattern, fix_norm, re.DOTALL)
        content = original_norm
        applied_any = False

        for search_text, replace_text in matches:
            # Strip trailing REPLACE marker if it leaked into replace_text
            replace_text = re.sub(r"\n>>>>>>> REPLACE.*", "", replace_text, flags=re.DOTALL)

            # a. Direct exact substring match
            if search_text in content:
                content = content.replace(search_text, replace_text, 1)
                applied_any = True
                continue

            # b. Strip boundary whitespace match
            s_stripped = search_text.strip()
            if s_stripped and s_stripped in content:
                content = content.replace(s_stripped, replace_text.strip(), 1)
                applied_any = True
                continue

            # c. Line-by-line whitespace-tolerant match
            s_lines = search_text.splitlines()
            while s_lines and not s_lines[0].strip():
                s_lines.pop(0)
            while s_lines and not s_lines[-1].strip():
                s_lines.pop()

            orig_lines = content.splitlines()
            n = len(s_lines)
            matched = False

            if n > 0 and len(orig_lines) >= n:
                for i in range(len(orig_lines) - n + 1):
                    window = orig_lines[i : i + n]
                    if all(w.strip() == s.strip() for w, s in zip(window, s_lines)):
                        # Preserve indentation delta
                        orig_indent = len(window[0]) - len(window[0].lstrip())
                        s_indent = len(s_lines[0]) - len(s_lines[0].lstrip())
                        delta = orig_indent - s_indent

                        adj_repl = []
                        for r_line in replace_text.splitlines():
                            if r_line.strip():
                                if delta > 0:
                                    adj_repl.append(" " * delta + r_line)
                                elif delta < 0 and r_line.startswith(" " * abs(delta)):
                                    adj_repl.append(r_line[abs(delta) :])
                                else:
                                    adj_repl.append(r_line)
                            else:
                                adj_repl.append("")

                        content = "\n".join(orig_lines[:i] + adj_repl + orig_lines[i + n :])
                        matched = True
                        applied_any = True
                        break

            if matched:
                continue

            # d. Function-level replacement fallback if search block or replace block defined a function
            func_m = re.search(r"^[ \t]*(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(", search_text, re.MULTILINE)
            if not func_m:
                func_m = re.search(r"^[ \t]*(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(", replace_text, re.MULTILINE)
            if func_m:
                func_name = func_m.group(1)
                # Matches def func_name(...): including multiline arguments and full indented body
                pattern_func = rf"(^[ \t]*(?:async\s+)?def\s+{re.escape(func_name)}\s*\([^)]*\):[ \t]*\n(?:[ \t]+.*?\n|\s*\n)*)"
                m_orig = re.search(pattern_func, content, re.MULTILINE | re.DOTALL)
                if m_orig:
                    content = content[:m_orig.start(1)] + replace_text.strip() + "\n\n" + content[m_orig.end(1):]
                    applied_any = True
                    logger.info(f"Applied function-level patch fallback for `{func_name}`")
                    continue

            logger.warning(f"Could not cleanly match search block:\n{search_text[:80]}")

        # If <<<<<<< SEARCH was present, NEVER return raw fix_block with conflict markers
        return content

    # 2. Guard against unintended full-file wiping of large files:
    # If the original file has > 60 lines and fix_block has < 60% of the lines
    if len(original_norm.splitlines()) > 60 and len(fix_norm.splitlines()) < len(original_norm.splitlines()) * 0.6:
        functions = re.findall(r"^[ \t]*(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(", fix_norm, re.MULTILINE)
        if functions:
            content = original_norm
            for func_name in functions:
                pattern = rf"(^[ \t]*(?:async\s+)?def\s+{re.escape(func_name)}\s*\([^)]*\):[ \t]*\n(?:[ \t]+.*?\n|\s*\n)*)"
                m_fix = re.search(pattern, fix_norm, re.MULTILINE | re.DOTALL)
                m_orig = re.search(pattern, content, re.MULTILINE | re.DOTALL)
                if m_fix and m_orig:
                    content = content[:m_orig.start(1)] + m_fix.group(1).strip() + "\n\n" + content[m_orig.end(1):]
                    logger.info(f"Safely updated function `{func_name}` in large file without truncating.")
            new_imports = [l for l in fix_norm.splitlines() if (l.startswith("import ") or l.startswith("from ")) and l not in content]
            if new_imports:
                content = "\n".join(new_imports) + "\n" + content
            return content
        else:
            logger.warning("Rejecting fix snippet that would truncate large repository file without SEARCH/REPLACE.")
            return original_norm

    # 3. Targeted function/method replacement for medium files
    func_match = re.search(r"^[ \t]*(?:async\s+)?def\s+([a-zA-Z0-9_]+)\s*\(", fix_norm, re.MULTILINE)
    if func_match and len(original_norm.splitlines()) > 25:
        func_name = func_match.group(1)
        pattern = rf"(^[ \t]*(?:async\s+)?def\s+{re.escape(func_name)}\s*\([^)]*\):[ \t]*\n(?:[ \t]+.*?\n|\s*\n)*)"
        m = re.search(pattern, original_norm, re.MULTILINE | re.DOTALL)
        if m:
            logger.info(f"Replacing standalone function `{func_name}` in original file.")
            return original_norm[:m.start(1)] + fix_norm.strip() + "\n\n" + original_norm[m.end(1):]

    # 4. Default: Full file replacement for small files
    return fix_norm

class NanoFixer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate_fix(
        self,
        plan: Dict[str, Any],
        target_files: Dict[str, str],
        latest_test_output: str,
        iteration: int = 1,
        previous_failure_details: Optional[str] = None,
        user_feedback: Optional[str] = None
    ) -> Dict[str, str]:
        """Generates the code fix using the fast Nano model and applies patches directly to target files."""

        files_context = []
        # Extract keywords from plan to focus window preview on relevant code region
        plan_text = f"{plan.get('issue_summary', '')} {plan.get('root_cause', '')} {plan.get('strategy', '')}".lower()
        terms = set(re.findall(r"[a-zA-Z]{4,}", plan_text))

        for path, content in target_files.items():
            # For large files, provide a targeted window in the prompt while preserving full content for patching
            if len(content) > 1800:
                lines = content.splitlines()
                matched_idx = -1
                for idx, line in enumerate(lines):
                    line_l = line.lower()
                    if any(t in line_l for t in terms):
                        matched_idx = idx
                        break
                if matched_idx != -1:
                    start_l = max(0, matched_idx - 25)
                    end_l = min(len(lines), matched_idx + 45)
                    preview = "\n".join(lines[start_l:end_l])
                else:
                    preview = content[:1800]
            else:
                preview = content

            files_context.append(f"### Target File: {path}\n```python\n{preview}\n```")
        files_text = "\n\n".join(files_context)

        feedback_section = ""
        if previous_failure_details:
            feedback_section += f"""\n### Iteration History & Failure Diagnosis:
The previous patch failed with the following remaining error:
```
{previous_failure_details[:800]}
```
CRITICAL: Do not repeat the identical fix. Address the remaining failing behavior.\n"""

        if user_feedback:
            feedback_section += f"""\n### Reviewer Feedback:
{user_feedback}\n"""

        prompt = f"""### Task (Fix Iteration #{iteration}):
Follow the Ultra Planner strategy to correct the failing code and make all tests pass.

### Ultra Plan:
- Issue: {plan.get('issue_summary', '')}
- Root Cause: {plan.get('root_cause', '')}
- Strategy: {plan.get('strategy', '')}
{feedback_section}
### Current Test Suite Output:
```
{latest_test_output[:800]}
```

### Files to Modify:
{files_text}

Output the fix inside ```python:path/to/file.py blocks using SEARCH/REPLACE or complete file code.
"""

        messages = [
            {"role": "system", "content": NANO_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        raw_response = await self.llm.call(
            messages=messages,
            role="nano",
            temperature=0.1
        )

        # Parse file blocks from output
        raw_blocks = self._parse_file_blocks(raw_response, list(target_files.keys()))

        # Apply patches directly so caller receives valid, patched file content
        patched_files: Dict[str, str] = {}
        for path, block_content in raw_blocks.items():
            orig_content = target_files.get(path)
            if orig_content is None:
                # Find matching target by basename
                target_key = self._find_matching_target(path, list(target_files.keys()))
                orig_content = target_files.get(target_key, "")
                if orig_content:
                    path = target_key

            if orig_content:
                patched_files[path] = apply_patch_or_content(orig_content, block_content)
            else:
                # New file or direct content
                patched_files[path] = block_content

        return patched_files

    def _parse_file_blocks(self, response_text: str, expected_files: List[str]) -> Dict[str, str]:
        """Extracts code blocks formatted in various common LLM output styles."""
        results: Dict[str, str] = {}
        clean_text = response_text.replace("\r\n", "\n")

        # Pattern 1: ```python:path/to/file.py\ncode\n(```|\Z)
        p1 = r"```(?:python:)?([a-zA-Z0-9_\-\./\\]+\.py)[ \t]*\n(.*?)(?:\n```|\Z)"
        for filepath, code in re.findall(p1, clean_text, re.DOTALL):
            filepath = filepath.strip()
            matched = self._find_matching_target(filepath, expected_files)
            results[matched or filepath] = code.strip()

        # Pattern 2: Header before block: e.g. ### File: path/to/file.py\n```python\ncode\n```
        if not results:
            p2 = r"(?:###|\*\*|File:)\s*`?([a-zA-Z0-9_\-\./\\]+\.py)`?.*?\n```(?:python)?\n(.*?)(?:\n```|\Z)"
            for filepath, code in re.findall(p2, clean_text, re.DOTALL):
                filepath = filepath.strip()
                matched = self._find_matching_target(filepath, expected_files)
                results[matched or filepath] = code.strip()

        # Pattern 3: Path inside code block on first line: ```python\n# path/to/file.py\ncode\n```
        if not results:
            p3 = r"```(?:python)?\n[#//]\s*([a-zA-Z0-9_\-\./\\]+\.py)[ \t]*\n(.*?)(?:\n```|\Z)"
            for filepath, code in re.findall(p3, clean_text, re.DOTALL):
                filepath = filepath.strip()
                matched = self._find_matching_target(filepath, expected_files)
                results[matched or filepath] = code.strip()

        # Pattern 4: Un-fenced SEARCH/REPLACE block
        if not results and "<<<<<<< SEARCH" in clean_text:
            search_match = re.search(r"(<<<<<<< SEARCH\s*\n.*?(?:\n>>>>>>> REPLACE|\Z))", clean_text, re.DOTALL)
            if search_match and expected_files:
                results[expected_files[0]] = search_match.group(1).strip()

        # Pattern 5: Generic python code block if expected_files has at least 1 entry
        if not results and expected_files:
            p5 = r"```(?:python)?\n(.*?)(?:\n```|\Z)"
            matches5 = re.findall(p5, clean_text, re.DOTALL)
            if matches5:
                results[expected_files[0]] = matches5[0].strip()

        return results

    def _find_matching_target(self, parsed_path: str, expected_files: List[str]) -> str:
        clean_p = parsed_path.strip().lstrip("./").lstrip("/")
        if clean_p.startswith("a/") or clean_p.startswith("b/"):
            clean_p = clean_p[2:]

        for expected in expected_files:
            clean_exp = expected.strip().lstrip("./").lstrip("/")
            if clean_p == clean_exp:
                return expected
            if clean_p.endswith("/" + clean_exp) or clean_exp.endswith("/" + clean_p):
                return expected
            if Path(clean_p).name == Path(clean_exp).name and Path(clean_p).name != "":
                return expected

        return parsed_path
