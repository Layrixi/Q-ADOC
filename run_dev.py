"""
run_dev.py

Starts the FastAPI backend and the Vite frontend dev server together, with
their output labeled and interleaved in one terminal, and a single Ctrl+C
cleanly stopping both.

Does NOT manage Ollama - that's expected to already be running natively on
your host (see README). This script does a quick check and warns you if it
doesn't look reachable, but won't block startup on it.

Run with the same Python environment (venv) that has your backend
dependencies installed:
    python run_dev.py
"""

import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
FRONTEND_DIR = PROJECT_ROOT / "frontend-react"
OLLAMA_HEALTH_URL = "http://localhost:11434"


def check_ollama():
    """Best-effort check - just a warning, never blocks startup."""
    try:
        urllib.request.urlopen(OLLAMA_HEALTH_URL, timeout=2)
        print("[launcher] Ollama looks reachable on localhost:11434.")
    except Exception:
        print(
            "[launcher] WARNING: couldn't reach Ollama on localhost:11434. "
            "Answers will fail until it's running (see README)."
        )


def stream_output(process: subprocess.Popen, label: str):
    """Reads a subprocess's stdout line by line and prints it with a label,
    so both processes' output can be interleaved in one terminal instead of
    needing two separate windows."""
    for line in iter(process.stdout.readline, ""):
        if line:
            print(f"[{label}] {line.rstrip()}")


def main():
    check_ollama()

    print("[launcher] Starting backend (uvicorn)...")
    # sys.executable ensures this uses whatever Python environment is
    # currently running this script (e.g. your venv), not just whatever
    # "python" resolves to on PATH - important so it picks up the same
    # installed packages (fastapi, uvicorn, etc.) automatically.
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    print("[launcher] Starting frontend (npm run dev)...")
    # shell=True on Windows is needed here because npm is actually npm.cmd,
    # and Popen won't resolve that correctly with a plain argument list.
    frontend = subprocess.Popen(
        "npm run dev",
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True,
    )

    # Each process's output is read on its own thread - otherwise reading
    # backend.stdout would block while frontend has output waiting (and
    # vice versa)
    backend_thread = threading.Thread(target=stream_output, args=(backend, "backend"), daemon=True)
    frontend_thread = threading.Thread(target=stream_output, args=(frontend, "frontend"), daemon=True)
    backend_thread.start()
    frontend_thread.start()

    print("[launcher] Both starting up. Backend: http://localhost:8000  Frontend: http://localhost:5173")
    print("[launcher] Press Ctrl+C to stop both.\n")

    try:
        while True:
            time.sleep(0.5)
            # If either process dies on its own (crash), stop the other
            # too rather than leaving one running uselessly.
            if backend.poll() is not None:
                print("[launcher] Backend exited unexpectedly - stopping frontend too.")
                break
            if frontend.poll() is not None:
                print("[launcher] Frontend exited unexpectedly - stopping backend too.")
                break
    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C received - shutting down both...")
    finally:
        for proc, name in [(backend, "backend"), (frontend, "frontend")]:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"[launcher] {name} didn't stop in time, killing it.")
                    proc.kill()
        print("[launcher] Done.")


if __name__ == "__main__":
    main()
