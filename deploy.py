import os
import subprocess
import sys

flyctl = r"C:\Users\ROY\.fly\bin\flyctl.exe"
cwd = r"C:\Users\ROY\studycompanion"

# Load access token
tok = None
with open(r"C:\Users\ROY\.fly\config.yml", "r") as f:
    for line in f:
        if line.startswith("access_token:"):
            tok = line.replace("access_token:", "").strip()
            break

if not tok:
    print("Error: access_token not found in config.yml")
    sys.exit(1)

env = os.environ.copy()
env["FLY_ACCESS_TOKEN"] = tok
env["FLY_API_TOKEN"] = tok

print("Deploying studycompanion-ai to Fly.io...")
res = subprocess.run(
    [flyctl, "deploy", "--config", "fly.toml", "--remote-only", "--ha=false", "--app", "studycompanion-ai"],
    cwd=cwd,
    capture_output=True,
    text=True,
    env=env
)

print("Deploy Return Code:", res.returncode)
print("Deploy Stdout:\n", res.stdout)
print("Deploy Stderr:\n", res.stderr)

res_status = subprocess.run(
    [flyctl, "status", "--app", "studycompanion-ai"],
    cwd=cwd,
    capture_output=True,
    text=True,
    env=env
)
print("App Status:\n", res_status.stdout)
