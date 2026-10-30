import logging
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional

from app import config
from app.agent.orchestrator import orchestrator
from app.demo.sample_repos import get_demo_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("safefix.main")

app = FastAPI(title="SafeFix PR-Fixer Agent", version="2.0.0")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

class StartJobRequest(BaseModel):
    issue_url: Optional[str] = None
    demo_id: Optional[str] = None
    custom_repo: Optional[str] = None
    custom_issue_title: Optional[str] = None
    custom_issue_body: Optional[str] = None
    github_token: Optional[str] = None
    provider: Optional[str] = "groq" # "groq" or "nemotron"
    api_key: Optional[str] = None
    ultra_model: Optional[str] = None
    nano_model: Optional[str] = None
    test_command: Optional[str] = None

class AdjustRequest(BaseModel):
    feedback: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "default_provider": config.DEFAULT_PROVIDER,
            "groq_ultra": config.GROQ_ULTRA_MODEL,
            "groq_nano": config.GROQ_NANO_MODEL,
            "nemotron_ultra": config.NEMOTRON_ULTRA_MODEL,
            "nemotron_nano": config.NEMOTRON_NANO_MODEL,
        }
    )

@app.get("/api/demos")
async def api_demos():
    return {"demos": get_demo_list()}

@app.post("/api/start")
async def api_start(req: StartJobRequest):
    if not req.issue_url and not req.demo_id and not req.custom_repo:
        raise HTTPException(status_code=400, detail="Must provide issue_url, demo_id, or custom_repo")

    session = orchestrator.create_session()
    
    await orchestrator.start_job(
        session_id=session.session_id,
        issue_url=req.issue_url,
        demo_id=req.demo_id,
        custom_repo=req.custom_repo,
        custom_issue_title=req.custom_issue_title,
        custom_issue_body=req.custom_issue_body,
        github_token=req.github_token or config.GITHUB_TOKEN,
        provider=req.provider or config.DEFAULT_PROVIDER,
        api_key=req.api_key or (config.NVIDIA_API_KEY if "nemotron" in (req.provider or "") else config.GROQ_API_KEY),
        ultra_model=req.ultra_model,
        nano_model=req.nano_model,
        test_command=req.test_command
    )

    return {
        "status": "started",
        "session_id": session.session_id,
        "websocket_url": f"/ws/{session.session_id}"
    }

@app.get("/api/session/{session_id}")
async def api_get_session(session_id: str):
    session = orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    env_dict = None
    if session.env_info:
        try:
            from dataclasses import asdict
            env_dict = asdict(session.env_info)
        except Exception:
            env_dict = str(session.env_info)

    return {
        "session_id": session.session_id,
        "state": session.state,
        "issue": session.issue_info,
        "plan": session.plan,
        "diff": session.diff_data,
        "baseline_test": session.baseline_test,
        "current_test": session.current_test,
        "targeted_test": session.targeted_test,
        "regression_test": session.regression_test,
        "pr_info": session.pr_info,
        "logs": session.logs,
        "iterations": session.iterations,
        "env_info": env_dict,
        "blocked_reason": session.blocked_reason,
        "discovered_tests": session.discovered_tests,
        "patch_history": session.patch_history
    }

@app.post("/api/approve/{session_id}")
async def api_approve(session_id: str):
    success = orchestrator.approve_review(session_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot approve: Session not waiting for review or not found")
    return {"status": "approved", "session_id": session_id}

@app.post("/api/adjust/{session_id}")
async def api_adjust(session_id: str, req: AdjustRequest):
    success = orchestrator.reject_or_adjust(session_id, req.feedback)
    if not success:
        raise HTTPException(status_code=400, detail="Session not found")
    return {"status": "adjusting", "session_id": session_id}

@app.get("/api/diff/{session_id}/download")
async def api_download_diff(session_id: str):
    session = orchestrator.get_session(session_id)
    if not session or not session.diff_data:
        raise HTTPException(status_code=404, detail="No diff available")
    
    diff_text = session.diff_data.get("diff_text", "")
    return PlainTextResponse(
        content=diff_text,
        headers={"Content-Disposition": f"attachment; filename=safefix_{session_id}.patch"}
    )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = orchestrator.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    session.websockets.add(websocket)
    try:
        env_dict = None
        if session.env_info:
            try:
                from dataclasses import asdict
                env_dict = asdict(session.env_info)
            except Exception:
                env_dict = str(session.env_info)

        # Send initial snapshot
        await websocket.send_json({
            "session_id": session.session_id,
            "event": "initial_state",
            "state": session.state,
            "data": {
                "session_id": session.session_id,
                "state": session.state,
                "issue": session.issue_info,
                "plan": session.plan,
                "diff": session.diff_data,
                "baseline_test": session.baseline_test,
                "current_test": session.current_test,
                "targeted_test": session.targeted_test,
                "regression_test": session.regression_test,
                "pr_info": session.pr_info,
                "logs": session.logs,
                "iterations": session.iterations,
                "env_info": env_dict,
                "blocked_reason": session.blocked_reason,
                "discovered_tests": session.discovered_tests,
                "patch_history": session.patch_history
            }
        })
        while True:
            # Keep socket open and accept any client messages
            data = await websocket.receive_text()
            # Can support ping/pong or interactive messages
    except WebSocketDisconnect:
        session.websockets.discard(websocket)
    except Exception:
        session.websockets.discard(websocket)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
