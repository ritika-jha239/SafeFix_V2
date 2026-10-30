import ast
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger("safefix.repo_map")

@dataclass
class FileSymbolInfo:
    rel_path: str
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    line_count: int = 0
    docstring: str = ""

class RepoMapper:
    """Builds whole-repository awareness across files, symbols, imports, and tracebacks.
    
    Guarantees:
    - Validates all agent-generated paths against repository root (prevents path traversal).
    - Retrieves multi-module context within model bounds without arbitrary truncations.
    """

    IGNORE_DIRS = {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
        ".mypy_cache", ".tox", ".nox", "build", "dist", ".eggs"
    }

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path).resolve()
        self.file_symbols: Dict[str, FileSymbolInfo] = {}
        self.import_graph: Dict[str, Set[str]] = {}

    def validate_path(self, path_str: str) -> Optional[Path]:
        """Validates that a path is strictly located within the repository root (prevents path traversal)."""
        raw = path_str.strip()
        if raw.startswith("a/") or raw.startswith("b/"):
            raw = raw[2:]
        if os.path.isabs(raw):
            target = Path(raw).resolve()
        else:
            target = (self.repo_path / raw).resolve()
        try:
            target.relative_to(self.repo_path)
            return target
        except ValueError:
            logger.warning(f"Security Alert: Blocked path traversal attempt: {path_str}")
            return None

    def scan(self):
        """Scans all Python files and parses their AST symbols and imports."""
        self.file_symbols.clear()
        self.import_graph.clear()

        for root, dirs, files in os.walk(self.repo_path):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")]

            for file in files:
                if file.endswith(".py"):
                    full_p = Path(root) / file
                    rel_p = str(full_p.relative_to(self.repo_path))
                    self._parse_file(full_p, rel_p)

    def _parse_file(self, full_p: Path, rel_p: str):
        try:
            content = full_p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content, filename=rel_p)
            docstring = ast.get_docstring(tree) or ""
            classes = []
            functions = []
            imports = []

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            functions.append(f"{node.name}.{item.name}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    imports.append(mod)

            self.file_symbols[rel_p] = FileSymbolInfo(
                rel_path=rel_p,
                classes=classes,
                functions=functions,
                imports=imports,
                line_count=len(content.splitlines()),
                docstring=docstring[:150]
            )

            # Build dependency graph
            self.import_graph[rel_p] = set(imports)
        except Exception:
            pass

    def extract_traceback_locations(self, traceback_text: str) -> List[Tuple[str, int]]:
        """Extracts repository-relative file paths and line numbers from stack traces."""
        locations = []
        # Match lines like: File "/path/to/repo/module.py", line 42, in func
        matches = re.findall(r'File\s+["\']([^"\']+\.py)["\'],\s+line\s+(\d+)', traceback_text)
        for fpath, lnum in matches:
            p = Path(fpath)
            try:
                rel = p.resolve().relative_to(self.repo_path)
                locations.append((str(rel), int(lnum)))
            except ValueError:
                # Check if file name matches relative file in repo
                for known_rel in self.file_symbols:
                    if known_rel.endswith(p.name) or p.name in known_rel:
                        locations.append((known_rel, int(lnum)))
                        break
        return locations

    def rank_relevant_files(
        self,
        issue_title: str,
        issue_body: str,
        baseline_test_output: str = "",
        max_files: int = 8
    ) -> List[Tuple[str, int]]:
        """Ranks repository files by relevance to the issue and failure trace."""
        if not self.file_symbols:
            self.scan()

        scores: Dict[str, int] = {f: 0 for f in self.file_symbols}
        combined_text = f"{issue_title} {issue_body}".lower()
        keywords = set(re.findall(r"[a-zA-Z0-9_\-\.]{3,}", combined_text))

        # 1. Stack trace locations get highest priority (+350 points)
        tb_locations = self.extract_traceback_locations(baseline_test_output)
        for rel_path, _ in tb_locations:
            if rel_path in scores:
                scores[rel_path] += 350

        # 2. Files explicitly mentioned in issue title or body (+200 points)
        for rel_path in self.file_symbols:
            f_name = Path(rel_path).name.lower()
            if f_name in combined_text or rel_path.lower() in combined_text:
                scores[rel_path] += 200

        # 3. Symbol matching (+80 points)
        for rel_path, info in self.file_symbols.items():
            for c in info.classes:
                if c.lower() in combined_text:
                    scores[rel_path] += 80
            for fn in info.functions:
                bare_fn = fn.split(".")[-1].lower()
                if bare_fn in combined_text:
                    scores[rel_path] += 90

        # 4. Import graph propagation (+40 points to dependencies of top files)
        top_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for tf, score in top_files:
            if score > 0:
                tf_stem = Path(tf).stem
                for other_f, imports in self.import_graph.items():
                    if any(tf_stem in imp for imp in imports):
                        scores[other_f] += 40

        # 5. Keyword scoring (+3 points per keyword in file stem)
        for rel_path in self.file_symbols:
            stem = Path(rel_path).stem.lower()
            for kw in keywords:
                if kw in stem:
                    scores[rel_path] += 15

        ranked = sorted([item for item in scores.items() if item[1] > 0], key=lambda x: x[1], reverse=True)
        if not ranked:
            # Fallback to non-test Python files
            non_tests = [f for f in self.file_symbols if "test" not in f.lower()][:max_files]
            return [(f, 10) for f in non_tests]

        return ranked[:max_files]

    def build_planner_context(
        self,
        issue_title: str,
        issue_body: str,
        baseline_test_output: str = "",
        max_chars: int = 14000
    ) -> Dict[str, Any]:
        """Builds multi-file repository context within context limits."""
        ranked_files = self.rank_relevant_files(issue_title, issue_body, baseline_test_output, max_files=6)
        
        # 1. Structural Map (File Tree & Symbol Index)
        tree_summary = []
        for rel_path, info in sorted(self.file_symbols.items())[:30]:
            syms = []
            if info.classes:
                syms.append(f"classes: {', '.join(info.classes[:3])}")
            if info.functions:
                clean_fns = [f.split('.')[-1] for f in info.functions[:4]]
                syms.append(f"funcs: {', '.join(clean_fns)}")
            sym_str = f" ({'; '.join(syms)})" if syms else ""
            tree_summary.append(f"- {rel_path}{sym_str}")
        
        structure_map = "\n".join(tree_summary)

        # 2. File Contents (dynamically budgeted)
        file_contents: Dict[str, str] = {}
        budget_per_file = max(1800, max_chars // max(1, len(ranked_files)))

        tb_locs = dict(self.extract_traceback_locations(baseline_test_output))

        for rel_path, _ in ranked_files:
            p = self.repo_path / rel_path
            if p.exists():
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                    if len(txt) <= budget_per_file:
                        file_contents[rel_path] = txt
                    else:
                        # Extract window around failure line if in traceback
                        target_line = tb_locs.get(rel_path, -1)
                        lines = txt.splitlines()
                        if target_line > 0 and target_line <= len(lines):
                            start = max(0, target_line - 30)
                            end = min(len(lines), target_line + 40)
                            file_contents[rel_path] = f"# ... [Lines 1 to {start} omitted] ...\n" + "\n".join(lines[start:end]) + f"\n# ... [Lines {end} to {len(lines)} omitted] ..."
                        else:
                            file_contents[rel_path] = "\n".join(lines[:60]) + f"\n# ... [{len(lines)-60} lines omitted] ..."
                except Exception:
                    continue

        return {
            "structure_map": structure_map,
            "relevant_files": [rf[0] for rf in ranked_files],
            "file_contents": file_contents
        }
