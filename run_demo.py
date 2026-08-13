"""SmartChunk Demo Launcher — Boots up the FastAPI backend and launches the browser."""

from __future__ import annotations

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.resolve()
VENV_DIR = ROOT_DIR / ".venv_demo"
DEMO_DIR = ROOT_DIR / "demo"

def log(msg: str):
    print(f"\033[95m[LAUNCHER]\033[0m {msg}")

def check_venv():
    """Verify virtual environment exists, creating it if necessary."""
    if not VENV_DIR.exists():
        log("Creating virtual environment '.venv_demo'...")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    
    # Get platform-specific executable paths
    if sys.platform == "win32":
        pip_exe = VENV_DIR / "Scripts" / "pip.exe"
        python_exe = VENV_DIR / "Scripts" / "python.exe"
    else:
        pip_exe = VENV_DIR / "bin" / "pip"
        python_exe = VENV_DIR / "bin" / "python"
        
    return python_exe, pip_exe

def install_dependencies(pip_exe: Path):
    """Ensure FastAPI, Uvicorn, and smartchunk extras are installed."""
    log("Checking & installing demo dependencies in virtual environment...")
    
    # Install fastapi, uvicorn, python-multipart and editable smartchunk with extras
    cmd = [
        str(pip_exe),
        "install",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "-e",
        str(ROOT_DIR) + "[all]"
    ]
    subprocess.run(cmd, check=True)

def start_server(python_exe: Path):
    """Start uvicorn server in uvicorn main loop."""
    log("Bootstrapping uvicorn server at http://127.0.0.1:8000 ...")
    
    # Start server in demo/ directory to let uvicorn resolve relative imports/static files
    os.chdir(DEMO_DIR)
    
    # Open browser slightly after server startup
    def open_browser():
        time.sleep(1.5)
        log("Opening web browser dashboard...")
        webbrowser.open("http://127.0.0.1:8000")
        
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        # Run Uvicorn directly from the virtual environment python interpreter
        cmd = [
            str(python_exe),
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000"
        ]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        log("Server stopped by user.")

def main():
    try:
        python_exe, pip_exe = check_venv()
        install_dependencies(pip_exe)
        start_server(python_exe)
    except Exception as e:
        print(f"\n\033[91m[ERROR]\033[0m Failed to start demo: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
