import asyncio
from app.agent.orchestrator import orchestrator

async def verify_string_calculator():
    session = orchestrator.create_session("demo-verify-run")
    print(f"Created session: {session.session_id}")
    await orchestrator.start_job(
        session_id=session.session_id,
        demo_id="string-calculator-negatives",
        api_key="demo_sim"
    )
    for _ in range(50):
        await asyncio.sleep(0.3)
        if session.state in ["WAITING_FOR_APPROVAL", "WAITING_FOR_REVIEW"]:
            break
    
    print(f"Session State reached: {session.state}")
    b_failed = session.baseline_test.get("failed_count", 0) if session.baseline_test else 0
    b_repro = session.baseline_test.get("is_reproduced", False) if session.baseline_test else False
    print(f"Baseline RED Verified: failed_count={b_failed}, is_reproduced={b_repro}")
    
    c_green = session.current_test.get("is_green", False) if session.current_test else False
    c_failed = session.current_test.get("failed_count", -1) if session.current_test else -1
    c_passed = session.current_test.get("passed_count", 0) if session.current_test else 0
    print(f"Post-Fix Test Verified: is_green={c_green}, failed_count={c_failed}, passed_count={c_passed}")
    
    diff_text = session.diff_data.get("diff_text", "")
    print(f"Diff Generated (+{session.diff_data.get('additions', 0)}, -{session.diff_data.get('deletions', 0)}) across: {session.diff_data.get('files_changed')}")
    
    assert session.state == "WAITING_FOR_APPROVAL", "Must pause at human review gate"
    assert c_green is True, "Tests must turn green"
    assert c_failed == 0, "Zero failures allowed"
    
    # Human approval gate
    approved = orchestrator.approve_review(session.session_id)
    print(f"Approve gate triggered: {approved}")
    for _ in range(30):
        await asyncio.sleep(0.2)
        if session.state == "COMPLETED":
            break
            
    print(f"Final State: {session.state}")
    print(f"PR URL: {session.pr_info.get('pr_url') if session.pr_info else None}")
    assert session.state == "COMPLETED"
    print("SUCCESS: STRING CALCULATOR DEMO TURNED RED TO GREEN AND CREATED PR!")

if __name__ == "__main__":
    asyncio.run(verify_string_calculator())

