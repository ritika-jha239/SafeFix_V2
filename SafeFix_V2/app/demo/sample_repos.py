from pathlib import Path
from typing import Dict, Any, List

SAMPLE_REPOS: Dict[str, Dict[str, Any]] = {
    "string-calculator-negatives": {
        "id": "string-calculator-negatives",
        "title": "Fix: Negatives not allowed should report all negative numbers in exception",
        "repo_name": "safefix-demo/string-calculator",
        "issue_number": 42,
        "issue_url": "https://github.com/safefix-demo/string-calculator/issues/42",
        "issue_body": """### Problem Description
When calling `add()` with multiple negative numbers like `add("1,-2,3,-4,-5")`, the function currently only lists the first negative number or raises without comma-separated list.

According to specification:
Calling `add` with negative numbers will throw an exception:
`negatives not allowed: -2, -4, -5`

### Failing Test
`test_multiple_negatives_raise_exception` in `tests/test_calculator.py` fails with:
`AssertionError: "negatives not allowed: -2, -4, -5" not found in "negatives not allowed: -2"`
""",
        "files": {
            "string_calculator.py": '''import re

def add(numbers: str) -> int:
    """Calculates sum of comma- or newline-separated numbers string.
    Raises ValueError with all negatives listed if any negative number is found.
    """
    if not numbers:
        return 0

    delimiter = ",|\\n"
    if numbers.startswith("//"):
        parts = numbers.split("\\n", 1)
        delimiter = re.escape(parts[0][2:])
        numbers = parts[1]

    tokens = [t for t in re.split(delimiter, numbers) if t]
    nums = [int(t) for t in tokens]

    # BUG: Only checks first negative number instead of collecting all
    for n in nums:
        if n < 0:
            raise ValueError(f"negatives not allowed: {n}")

    return sum(n for n in nums if n <= 1000)
''',
            "tests/test_calculator.py": '''import pytest
from string_calculator import add

def test_empty_string_returns_zero():
    assert add("") == 0

def test_single_and_two_numbers():
    assert add("1") == 1
    assert add("1,2") == 3

def test_newline_delimiter():
    assert add("1\\n2,3") == 6

def test_multiple_negatives_raise_exception():
    with pytest.raises(ValueError) as excinfo:
        add("1,-2,3,-4,-5")
    assert "negatives not allowed: -2, -4, -5" in str(excinfo.value)
'''
        },
        "test_command": "pytest -v tests/test_calculator.py"
    },
    "jwt-auth-expiration": {
        "id": "jwt-auth-expiration",
        "title": "Bug: Token expiry calculation fails on leeway comparison",
        "repo_name": "safefix-demo/jwt-auth-service",
        "issue_number": 88,
        "issue_url": "https://github.com/safefix-demo/jwt-auth-service/issues/88",
        "issue_body": """### Issue Description
Tokens within the allowable leeway window (e.g. 30 seconds clock drift) are being falsely marked as expired.
In `auth_validator.py`, the expiration check does not apply `leeway_seconds` properly when testing `current_time >= exp`.

### Expected
`validate_token_expiry(payload, leeway_seconds=30)` should return `True` if current time is within `exp + 30`.
""",
        "files": {
            "auth_validator.py": '''import time

def validate_token_expiry(payload: dict, leeway_seconds: int = 0) -> bool:
    """Validates JWT expiration timestamp with leeway window."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid payload format")

    exp = payload.get("exp")
    if exp is None:
        raise ValueError("Missing 'exp' claim in token")

    current_time = time.time()

    # BUG: Subtracted leeway instead of adding it, expiring tokens prematurely
    if current_time >= (exp - leeway_seconds):
        return False

    return True
''',
            "tests/test_auth.py": '''import time
import pytest
from auth_validator import validate_token_expiry

def test_future_token_is_valid():
    future = time.time() + 3600
    assert validate_token_expiry({"exp": future}) is True

def test_past_token_is_expired():
    past = time.time() - 3600
    assert validate_token_expiry({"exp": past}) is False

def test_token_within_leeway_window_is_valid():
    # Expired 10 seconds ago, but leeway is 30 seconds -> Should be valid!
    just_expired = time.time() - 10
    assert validate_token_expiry({"exp": just_expired}, leeway_seconds=30) is True
'''
        },
        "test_command": "pytest -v tests/test_auth.py"
    },
    "slug-unicode-generator": {
        "id": "slug-unicode-generator",
        "title": "Unicode characters cause slug generator to produce empty or truncated slugs",
        "repo_name": "safefix-demo/slug-generator",
        "issue_number": 15,
        "issue_url": "https://github.com/safefix-demo/slug-generator/issues/15",
        "issue_body": """### Description
When given unicode strings like `"Crème Brûlée & Café"`, `generate_slug()` strips non-ascii characters directly instead of decomposing accented letters into ascii (e.g. 'e' instead of ''), producing `"crm-brl-caf"`.

Expected: `"creme-brulee-cafe"`
""",
        "files": {
            "slugify.py": '''import re

def generate_slug(text: str) -> str:
    """Converts a string into an SEO-friendly URL slug."""
    if not text:
        return ""

    # BUG: Does not normalize accents with unicodedata NFKD before regex
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\\s-]", "", text)
    text = re.sub(r"[\\s-]+", "-", text)
    return text.strip("-")
''',
            "tests/test_slug.py": '''from slugify import generate_slug

def test_basic_slug():
    assert generate_slug("Hello World") == "hello-world"

def test_unicode_accents_slug():
    # Should normalize accents to plain ascii
    assert generate_slug("Crème Brûlée") == "creme-brulee"

def test_special_characters_slug():
    assert generate_slug("Fast & Furious: 2026") == "fast-furious-2026"
'''
        },
        "test_command": "pytest -v tests/test_slug.py"
    }
}

def create_sample_repo(demo_id: str, target_dir: Path) -> Dict[str, Any]:
    """Populates target directory with files for the chosen sample repository."""
    if demo_id not in SAMPLE_REPOS:
        demo_id = "string-calculator-negatives"

    demo_data = SAMPLE_REPOS[demo_id]
    target_dir.mkdir(parents=True, exist_ok=True)

    for relative_path, content in demo_data["files"].items():
        file_path = target_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    return demo_data

def get_demo_list() -> List[Dict[str, Any]]:
    return [
        {
            "id": k,
            "title": v["title"],
            "repo_name": v["repo_name"],
            "issue_number": v["issue_number"],
            "issue_url": v["issue_url"],
            "issue_body": v["issue_body"],
            "test_command": v["test_command"]
        }
        for k, v in SAMPLE_REPOS.items()
    ]

