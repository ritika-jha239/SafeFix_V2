import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
# Keep sandboxes in /tmp so file writes never trigger any workspace watchers
SANDBOX_DIR = Path("/tmp/safefix_sandboxes")
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

# Default LLM Providers: "groq" or "nemotron"
DEFAULT_PROVIDER = os.getenv("SAFEFIX_PROVIDER", "groq")

# Groq Defaults (Configured with user's enabled qwen/qwen3.8-27b model)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_ULTRA_MODEL = os.getenv("GROQ_ULTRA_MODEL", "qwen/qwen3.8-27b")
GROQ_NANO_MODEL = os.getenv("GROQ_NANO_MODEL", "qwen/qwen3.8-27b")

# NVIDIA Token Factory / Nemotron Defaults
# Strictly local; can be provided via local .env (in .gitignore) or in-browser UI
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NEMOTRON_ULTRA_MODEL = os.getenv("NEMOTRON_ULTRA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
NEMOTRON_NANO_MODEL = os.getenv("NEMOTRON_NANO_MODEL", "nvidia/nemotron-mini-4b-instruct")

# GitHub Token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Execution parameters
MAX_FIX_ITERATIONS = int(os.getenv("MAX_FIX_ITERATIONS", "3"))
TEST_TIMEOUT_SECONDS = int(os.getenv("TEST_TIMEOUT_SECONDS", "35"))
