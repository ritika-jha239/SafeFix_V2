# 🛠️ SafeFix: Autonomous PR-Fixer Agent

> **Point it at a failing GitHub repository or issue. An Ultra + Nano reasoning agent reads the code, diagnoses the root cause, writes a patch, runs tests in a sandbox, loops until the red test suite turns glowing green, and halts at an interactive review gate ("Ask Button") before opening a GitHub Pull Request.**

---

## 🌟 Highlights

- **Dual-Model Hierarchical Agent Architecture**:
  - 🧠 **Ultra Planner** (*Nemotron-70B* or *Groq LLaMA-3.3-70B*): Deep reasoning agent that analyzes GitHub issues, inspects repo file trees, tracebacks, and formulates an ironclad Root Cause Analysis & Fix Strategy.
  - ⚡ **Nano Fixer** (*Nemotron-Mini* or *Groq LLaMA-3.1-8B*): Ultra-fast coder agent that generates precise file modifications and iterates rapidly in the test-fix loop.
- **Dynamic Red ➔ Green Test Suite Transformation**:
  - Live sandbox executor captures real-time `pytest` or `unittest` output.
  - Shows instant status shift on screen from **❌ RED (Failing)** to **✅ GREEN (Passing)**.
- **Interactive Human Review Gate (The Ask Button)**:
  - When tests pass, the agent halts and surfaces a unified Git Diff review, lines changed (+/-), and test reports.
  - Asks the user: *"All tests passed! Ready to push changes and open PR?"*
  - Requires user approval via **"Approve & Open GitHub PR"** button before any remote push.
- **Automated Pull Request Creation**:
  - Autonomous Git branching (`safefix/fix-issue-<id>`).
  - Conventional commit creation.
  - Automated PR opening with root cause analysis, strategy, Red ➔ Green test logs, and `Fixes #<id>`.
- **Pluggable LLM Backends**:
  - **Groq API**: Lightning-fast inference for rapid testing and iteration.
  - **NVIDIA NIM / Token Factory Nemotron**: High-capacity enterprise reasoning models (`nvidia/llama-3.1-nemotron-70b-instruct`).
- **Built-in One-Click Demo Scenarios**:
  - 🔴 *Demo 1: String Calculator (Negative Number Exception Formatting)*
  - 🔴 *Demo 2: JWT Auth Validator (Clock Drift & Leeway Bug)*
  - 🔴 *Demo 3: URL Slug Generator (Accented Unicode Normalization)*

---

## 🏛️ Architecture Overview

```
                                  +------------------------------------+
                                  |         GitHub Issue URL           |
                                  | (or Built-in One-Click Demo)       |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |        SafeFix Ingestion           |
                                  |   Clone Repo & Establish Sandbox   |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |     Baseline Sandbox Execution     |
                                  |      Run Test Suite (RED ❌)       |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |       Ultra Planner (Nemotron)     |
                                  |   Issue Analysis & Strategy Plan   |
                                  +-----------------+------------------+
                                                    |
                                                    v
                        +---------------------------+---------------------------+
                        |                                                       |
                        |      Nano Fixer (Fast Code Patch Generator)           |
                        |                        |                              |
                        |                        v                              |
                        |              Apply Patch to Disk                      |
                        |                        |                              |
                        |                        v                              |
                        |             Run Sandbox Test Suite                    |
                        |                        |                              |
                        |               [Tests Still Failing?]                  |
                        |               /                    \                  |
                        |         Yes  /                      \  No             |
                        +-------------+                        |                |
                                                               v                |
                                             +----------------------------------+
                                             |     Test Suite Turns GREEN ✅     |
                                             +-----------------+----------------+
                                                               |
                                                               v
                                             +----------------------------------+
                                             |   INTERACTIVE HUMAN REVIEW GATE  |
                                             |  - Diff Viewer & Lines Added/Del |
                                             |  - Test Scorecard & Commit Msg   |
                                             |  - [Approve & Push to GitHub]    |
                                             +-----------------+----------------+
                                                               |
                                                  User Clicks [Approve]
                                                               |
                                                               v
                                             +----------------------------------+
                                             |   Branch, Commit & Open PR 🚀    |
                                             |  https://github.com/.../pull/..  |
                                             +----------------------------------+
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Credentials (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Or configure keys directly in the web UI Settings drawer:
- **Groq API Key**: Get from [console.groq.com](https://console.groq.com)
- **NVIDIA NIM API Key**: Get from [build.nvidia.com](https://build.nvidia.com)
- **GitHub PAT**: Needed only if you wish to push real branches & open PRs on GitHub.

*(Note: SafeFix includes high-fidelity synthetic demo generators so you can test all features and the full Red ➔ Green ➔ Review ➔ PR workflow immediately out of the box even without API keys!)*

### 4. Launch SafeFix Server
```bash
python3 -m app.main
```
Or with uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```
Navigate to:
```
http://localhost:8001
```

---

## 🧪 Running Automated Tests

Run the full pytest suite:
```bash
PYTHONPATH=. .venv/bin/pytest -v tests/
```

Test coverage includes:
- Isolated test runner stdout/stderr parsing and exit code tracking.
- Baseline failure verification (RED state).
- Ultra Planner diagnostic JSON extraction.
- Nano Fixer code patch application.
- Sandbox test re-execution and GREEN confirmation.
- Unified Git diff generation.
- Full asynchronous orchestrator lifecycle with review gate pause and resume.
- All three built-in demo scenarios.
- FastAPI REST endpoints.

---

## 📡 API & WebSocket Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | SafeFix Interactive Web Dashboard |
| `GET` | `/api/demos` | List built-in demo scenarios |
| `POST` | `/api/start` | Launch PR-Fixer on an issue URL or demo |
| `GET` | `/api/session/{id}` | Fetch current session state, plan, diff, test results |
| `POST` | `/api/approve/{id}` | **The Ask Button**: Approve review and trigger GitHub PR |
| `POST` | `/api/adjust/{id}` | Provide feedback / request adjustments to fix |
| `GET` | `/api/diff/{id}/download` | Download unified git diff `.patch` file |
| `WS` | `/ws/{id}` | Live streaming WebSocket for terminal output, state transitions & alerts |

---

## 💡 Tech Stack
- **Backend**: FastAPI, Asyncio, Subprocess Sandboxing, GitPython, Httpx, Pydantic
- **Frontend**: Responsive Tailwind CSS, WebSockets, FontAwesome, Monospace Terminal & Diff Viewers
- **AI Models**: Groq (LLaMA-3.3 70B & LLaMA-3.1 8B), NVIDIA NIM Nemotron (Ultra & Mini/Nano)

