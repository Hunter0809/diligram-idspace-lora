"""Run task-scoped scripts through the historical 4090 Windows/WSL host.

The host and account are fixed for this project. The password is read from the
user-scoped ``diligram`` environment variable at runtime and is never written
to the repository.
"""
import argparse
import base64
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid


def _diligram_password() -> str:
    password = next((value for key, value in os.environ.items() if key.lower() == "diligram"), "")
    if password:
        return password
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                value, _ = winreg.QueryValueEx(key, "diligram")
                return str(value)
        except (FileNotFoundError, OSError):
            pass
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--platform", choices=("linux", "windows"), default="linux")
    args = parser.parse_args()
    contents = args.script.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    remote_name = "mineru_task_" + uuid.uuid4().hex + (".sh" if args.platform == "linux" else ".ps1")
    if args.platform == "linux":
        source = f"wsl.exe -d Ubuntu -- bash '/mnt/c/Users/mr jiang/{remote_name}'; $code=$LASTEXITCODE"
    else:
        source = f"& (Join-Path $env:USERPROFILE '{remote_name}'); $code=$LASTEXITCODE"
    source = ("$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
              + source + f"; Remove-Item -LiteralPath (Join-Path $env:USERPROFILE '{remote_name}'); exit $code")
    encoded = base64.b64encode(source.encode("utf-16le")).decode()
    helper = Path(os.environ["USERPROFILE"]) / ".codex/skills/hla-remote-ssh/scripts/hla_remote.py"
    env = os.environ.copy()
    password = _diligram_password()
    auth_home = Path(tempfile.mkdtemp(prefix="mineru_ssh_home_"))
    env.update({"HLA_SSH_HOST": "100.107.179.116", "HLA_SSH_PORT": "22",
                "HLA_SSH_USER": "mr jiang", "HLA_SSH_RETRIES": "1",
                "HLA_REMOTE_ROOT": "C:/Users/mr jiang",
                "HLA_SSH_KEY": "", "HOME": str(auth_home),
                "USERPROFILE": str(auth_home),
                "HLA_SSH_PASSWORD": password,
                "PYTHONIOENCODING": "utf-8"})
    try:
        with tempfile.NamedTemporaryFile(dir=args.script.resolve().parent, prefix=".ssh_upload_", delete=False) as prepared:
            prepared.write(contents)
            prepared_path = Path(prepared.name)
        try:
            uploaded = subprocess.call([sys.executable, str(helper), "upload", str(prepared_path), remote_name], env=env)
        finally:
            prepared_path.unlink()
        if uploaded:
            return uploaded
        return subprocess.call([sys.executable, str(helper), "run",
                                "powershell -NoProfile -EncodedCommand " + encoded], env=env)
    finally:
        shutil.rmtree(auth_home, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
