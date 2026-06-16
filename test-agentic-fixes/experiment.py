import csv

prompt_short = """
%s,%s
Fix this issue.
"""

prompt_long = """
You are an expert developer. You must fix a static analysis issue.

### Input
- **Message:** %s
- **Location:** %s

### Critical Thinking Process (Do not skip)
Before writing the fix, think through these steps:
1. Is this a genuine issue, or a false positive?
2. Can it be fixed without breaking existing logic?
3. What is the safest, cleanest way to fix it?

After reasoning and fixing the issue, you must provide a short summary of your conclusion.
"""


def format_issue_message(finding: dict) -> str:
    return f'{finding['Rule Key']} - {finding['Message']}'


def format_issue_location(finding: dict) -> str:
    return f'{finding['File Name']}:{finding['File Line']}'


def format_prompt(prompt: str, finding) -> str:
    return prompt.format(format_issue_message(finding), format_issue_location(finding))


def prepare_env(model: str, prompt_type: str) -> str:
    """
    TODO :
     - create empty dir for experiment
     - copy file_to_analyze.py into it to not alter original version
     - run fixing on the fresh file and output in the fresh dir
    """
    pass


def fix_with_agent(model: str, env: str):
    """
    TODO :
    - call `claude -c ...`
    - support : claude, codex, claude or codex with gemma4 (ollama connection mode)
    """
    pass


def post_analyze(env: str) -> dict:
    """todo : trigger analysis on fixed code, collect outputs, collect fixed python code"""
    pass


def main():
    with open('open_findings_on_overall_code.csv', 'r') as findingsFile:
        findingsReader = csv.DictReader(findingsFile)
        for finding in findingsReader:
            print(finding)
            print(format_issue_message(finding))


if __name__ == '__main__':
    main()
