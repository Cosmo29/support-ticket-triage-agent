"""
Unified Orchestrator and CLI Launcher for Support Ticket Triage Agent.
Allows launching the FastAPI server, Streamlit dashboard, eval harness, or test suite.
"""

import sys
import os
import argparse
import subprocess
import signal
import time

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_server():
    import uvicorn
    from app.config import settings
    print(f"🚀 Launching FastAPI Backend on http://{settings.HOST}:{settings.PORT}")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)

def run_dashboard():
    from app.config import settings
    print(f"🛡️ Launching Streamlit Dashboard on http://localhost:{settings.STREAMLIT_PORT}")
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "triage_dashboard.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.port", str(settings.STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])

def run_both():
    from app.config import settings
    print("=" * 70)
    print("🚀 Launching AI Support Ticket Triage Agent System (FastAPI + Streamlit)")
    print(f"  • FastAPI Backend:     http://localhost:{settings.PORT}")
    print(f"  • Streamlit Dashboard: http://localhost:{settings.STREAMLIT_PORT}")
    print(f"  • Swagger Docs:        http://localhost:{settings.PORT}/docs")
    print("=" * 70)

    # Launch FastAPI as subprocess
    proc_api = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", settings.HOST,
        "--port", str(settings.PORT)
    ])

    time.sleep(1.5)

    # Launch Streamlit as subprocess
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "triage_dashboard.py")
    proc_dash = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", dashboard_path,
        "--server.port", str(settings.STREAMLIT_PORT),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])

    def signal_handler(sig, frame):
        print("\nGracefully shutting down services...")
        proc_api.terminate()
        proc_dash.terminate()
        proc_api.wait()
        proc_dash.wait()
        print("All processes stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        proc_api.wait()
        proc_dash.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)

def main():
    parser = argparse.ArgumentParser(description="Support Ticket Triage Agent CLI")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI REST backend")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit triage dashboard")
    parser.add_argument("--all", action="store_true", help="Launch both backend and dashboard concurrently")
    parser.add_argument("--eval", action="store_true", help="Execute model benchmark evaluation harness")
    parser.add_argument("--test", action="store_true", help="Run test suite")

    args = parser.parse_args()

    if args.server:
        run_server()
    elif args.dashboard:
        run_dashboard()
    elif args.eval:
        from evals.eval_harness import run_evaluation
        run_evaluation()
    elif args.test:
        from run_tests import run_all_tests
        run_all_tests()
    else:
        # Default action
        run_both()

if __name__ == "__main__":
    main()
