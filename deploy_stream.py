import os
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

flyctl = r"C:\Users\ROY\.fly\bin\flyctl.exe"
cwd = r"C:\Users\ROY\studycompanion"

tok = None
with open(r"C:\Users\ROY\.fly\config.yml", "r") as f:
    for line in f:
        if line.startswith("access_token:"):
            tok = line.replace("access_token:", "").strip()
            break

env = os.environ.copy()
env["FLY_ACCESS_TOKEN"] = tok
env["FLY_API_TOKEN"] = tok

print("Starting streaming deploy to Fly.io...")
process = subprocess.Popen(
    [flyctl, "deploy", "--config", "fly.toml", "--remote-only", "--ha=false", "--app", "studycompanion-ai", "--yes"],
    cwd=cwd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
    env=env
)

for line in iter(process.stdout.readline, ''):
    print(line, end='', flush=True)

process.stdout.close()
return_code = process.wait()
print(f"\nFinished with code {return_code}")
