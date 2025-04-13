import subprocess
import time

def run_backend():
    return subprocess.Popen(["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"])

def run_frontend():
    return subprocess.Popen(["streamlit", "run", "frontend.py"])

if __name__ == "__main__":
    print("Starting backend (Uvicorn)...")
    backend_proc = run_backend()
    time.sleep(2)  # Give backend time to boot up

    print("Starting frontend (Streamlit)...")
    frontend_proc = run_frontend()

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()