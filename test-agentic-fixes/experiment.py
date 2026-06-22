import csv
import json
import shutil
import subprocess
import tempfile
import os
import time
from pathlib import Path

import requests

prompt_short = """{} {}
Fix this issue."""

prompt_long = """You are an expert developer. You must fix a static analysis issue.

### Input
- **Message:** {}
- **Location:** {}

### Critical Thinking Process (Do not skip)
Before writing the fix, think through these steps:
1. Is this a genuine issue, or a false positive?
2. Can it be fixed without breaking existing logic?
3. What is the safest, cleanest way to fix it?

After reasoning and fixing the issue, you must provide a short summary of your conclusion."""

prompts = {
    # 'short': prompt_short,
    'long': prompt_long,
}

CLAUDE = 'claude'
CODEX = 'codex'
GEMMA4 = 'gemma4'  # Small model to challenge the prompts
models = [
    # CLAUDE,
    # CODEX,
    GEMMA4
]


def format_issue_message(finding: dict) -> str:
    return f'{finding['Rule Key']} - {finding['Message']}'


def format_issue_location(finding: dict) -> str:
    return f'{finding['File Name']}:{finding['File Line']}'


def format_prompt(prompt: str, finding) -> str:
    prompt = prompt.format(format_issue_message(finding), format_issue_location(finding))
    prompt = prompt.replace('"', "'")
    return f'"{prompt}"'


def prepare_env() -> tempfile.TemporaryDirectory:
    """
    Creates an empty temporary directory for the experiment.
    Copies both 'file_to_analyze.py' and 'analyze.sh' into it
    and ensures 'analyze.sh' is executable.

    Returns:
        tempfile.TemporaryDirectory: The object managing the isolated workspace.
    """
    # 1. Create the unique temporary directory
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="experiment_")
    env_path = tmp_dir_obj.name

    required_files = ["file_to_analyze.py", "analyze.sh"]

    # 2. Copy both files into the sandbox
    for file_name in required_files:
        if os.path.exists(file_name):
            shutil.copy(file_name, os.path.join(env_path, file_name))
        else:
            # Clean up the directory before raising an error to avoid leaving junk
            tmp_dir_obj.cleanup()
            raise FileNotFoundError(f"Critical error: '{file_name}' must exist in the current working directory.")

    # 3. Ensure analyze.sh has execute permissions inside the temp folder
    temp_script_path = os.path.join(env_path, "analyze.sh")
    # Fetch current permissions, then binary OR them with the user-execute bit
    current_mode = os.stat(temp_script_path).st_mode
    os.chmod(temp_script_path, current_mode | 0o111)

    return tmp_dir_obj


def fix_with_agent(model: str, prompt: str, env: tempfile.TemporaryDirectory):
    """
    Invokes the requested AI engineering tool or local model.

    Args:
        model (str): Engine selector ('claude', 'codex', or 'gemma4').
        prompt (str): Formatted context/instructions for fixing the issue.
        env (tempfile.TemporaryDirectory): The active temporary sandbox object.
    """
    # Extract the absolute path string from the tempfile object
    env_path = env.name

    if model.lower() == "claude":
        # Target Claude Sonnet 4.7 ecosystem configuration
        cmd = ["claude", "--permission-mode", "acceptEdits", "-p", prompt]

    elif model.lower() == "codex":
        # Launch local Codex via Ollama runtime architecture inside the sandbox context
        cmd = ["codex", "exec", "--sandbox", "workspace-write", "--skip-git-repo-check", prompt]

    elif model.lower() == "gemma4":
        cmd = ["codex", "--oss", "-m", "gemma4:26b-mlx", "exec",
               "--skip-git-repo-check", "--sandbox", "workspace-write", "--local-provider=ollama", prompt]

    else:
        raise ValueError(f"Unknown or unsupported model configuration: {model}")

    print(f"\n[Agent Invocation]: Deploying engine matching '{model}' details...")
    print(f"Command payload -> {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            cwd=env_path,
            env=os.environ,
            stdout=None,
            stderr=None,
            stdin=subprocess.DEVNULL,
            text=True,
            check=True,
            timeout=300
        )
        print(f"\n[{model} Response Log]: Execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️  Error: {model} pipeline exited abnormally with code {e.returncode}.")
    except subprocess.TimeoutExpired as e:
        print(f"\n⚠️  Error: {model} pipeline timed out after {e.timeout} seconds.")
    except Exception as e:
        print(f"\n⚠️  Unexpected error during {model} execution: {e}")


def fetch_findings_from_sonarqube(server_url: str, project_key: str) -> list:
    """Fetches open issues dynamically using the SonarQube Web API.

    Pulls the authentication token directly from the SONAR_TOKEN environment
    variable.
    """
    # Retrieve the token from the environment
    token = os.environ.get("SONAR_TOKEN")

    if not token:
        raise ValueError(
            "Environment variable 'SONAR_TOKEN' is not set. Please set it before running the script."
        )

    # The dedicated API endpoint for pulling issues/findings
    api_url = f"{server_url.rstrip('/')}/api/issues/search"

    # Target specific project, filter for open issues, and maximize page size
    params = {
        "componentKeys": project_key,
        "statuses": "OPEN",
        "ps": 500,  # Page size (max allowed by SonarQube is 500)
    }

    # Authenticate via Bearer Token
    headers = {"Authorization": f"Bearer {token}"}

    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()

        # This matches the structure returned by SonarQube's API response
        return data.get("issues", [])

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch issues from SonarQube API: {e}")
        return []


def analyze(env: tempfile.TemporaryDirectory) -> dict:
    """
    Triggers analysis on the fixed code inside the isolated temp folder,
    collects terminal execution output, and captures the updated python code.

    Args:
        env (tempfile.TemporaryDirectory): The active temporary directory object.

    Returns:
        dict: A findings data dictionary containing the logs and the rewritten source code.
    """
    # Extract the absolute path string from the tempfile object
    env_path = env.name

    target_file = os.path.join(env_path, "file_to_analyze.py")
    script_file = os.path.join(env_path, "analyze.sh")

    # 1. Trigger the analysis script local to the sandbox
    if os.path.exists(script_file):
        try:
            # We use cwd=env_path so analyze.sh executes within the temp folder context
            result = subprocess.run(
                ["./analyze.sh", "file_to_analyze.py"],
                cwd=env_path,
                capture_output=True,
                text=True,
                check=True
            )
            analysis_output = result.stdout
        except subprocess.CalledProcessError as e:
            # Captures output even if the script returns a non-zero exit code (e.g., failed tests)
            analysis_output = f"Analysis script execution terminated.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
    else:
        analysis_output = f"Error: analyze.sh not found inside sandbox environment ({env_path})."

    # 2. Collect the modified python code state
    fixed_code = ""
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            fixed_code = f.read()

    # 3. Return the packaged findings/results dictionary
    return {
        "status": "success" if "error" not in analysis_output.lower() else "failed",
        "analysis_output": analysis_output,
        "fixed_code": fixed_code
    }

def restart_ollama_server():
    print("[System]: Hard resetting Ollama Server...")

    # 1. Kill any existing server processes running on the machine
    # For Linux/macOS:
    subprocess.run(["pkill", "-f", "ollama serve"], stderr=subprocess.DEVNULL)
    # (If on Windows: subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"]))

    time.sleep(1) # Give the system a brief moment to release OS ports

    # 2. Spin the serve command back up as a background process
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL
    )

    time.sleep(3) # Give the server a few seconds to boot up before sending requests

def experiment(findings, model, prompt) -> tuple:
    env = prepare_env()
    print(env.name)
    for finding in findings:
        if model.lower() == "gemma4":
            restart_ollama_server()
        fix_with_agent(model=model, prompt=format_prompt(prompt, finding), env=env)
    result = analyze(env=env)
    print("Analysis result :", result)
    result = fetch_findings_from_sonarqube("http://localhost:9000", "quick-fixes-agent-integration")
    print(len(result), result)
    return result, env


def run_all(findings):
    for model in models:
        for prompt_type, prompt in prompts.items():
            output_folder = Path(f"experiment_results/{model}/{prompt_type}")
            output_folder.mkdir(parents=True, exist_ok=True)
            print(f"\n[Experiment]: Running with model '{model}' and prompt type '{prompt_type}'...")
            new_findings, tempDirectory = experiment(findings, model=model, prompt=prompt)
            with open(output_folder / "findings.json", "w") as f:
                json.dump(new_findings, f)
            shutil.copy(tempDirectory.name + "/file_to_analyze.py", output_folder / "file_to_analyze.py")


def main():
    # print(fetch_findings_from_sonarqube("http://localhost:9000", "quick-fixes-agent-integration"))
    # return
    with open('open_findings_on_overall_code.csv', 'r') as findingsFile:
        findings_reader = csv.DictReader(findingsFile)
        findings = list(findings_reader)
        run_all(findings)


if __name__ == '__main__':
    main()
