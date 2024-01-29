import ast
import concurrent.futures
import contextlib
import glob
import json
import os
import pathlib
import rich
import rich.progress
import subprocess
import sys
import textwrap
from pytype.config import Options
from pytype.tools.annotate_ast import annotate_ast
from typing import TypedDict, Sequence
from typing_extensions import NotRequired


def run_shell_command(command: Sequence[str], ok_codes: tuple[int, ...] = (0,), timeout=1200) \
        -> subprocess.CompletedProcess | None:
    """
    Run a shell command via subprocess.run() call.
    Args:
    command: command and options as list of string (as accepted by subprocess.run())
    ok_codes: the return codes considered as OK. If the actuaL return code is not in this list, an exception is raised.
Returns:
    the CompletedProcess object
    """
    try:
        process = subprocess.run(command, timeout=timeout)
        if process.returncode not in ok_codes:
            err_msg = process.stderr.decode("utf-8")
            raise RuntimeError(f"Calling \"{' '.join(command)}\" returns non-zero code {process.returncode} with the "
                               f"following stderr: {err_msg}")
        return process
    except subprocess.SubprocessError:
        return None

def get_files_in_dir(dir_path: str, file_extensions: tuple[str, ...] = (".py",)) -> list[str]:
    """
    Return (relative paths) of files in a directory. Possible to filter according to the file extensions
    Args:
        dir_path: the directory path
        file_extensions: (optionally) tuple of file extensions, e.g. (".py", ".pyi").
                         Default to empty tuple (considering all files)
    Returns: a list of relative file paths.
    """
    if isinstance(file_extensions, str):
        file_extensions = (file_extensions,)
    file_extensions = [f if f.startswith(".") else "."+f for f in file_extensions]
    relative_paths: list[str]
    if os.path.isdir(dir_path):
        relative_paths = glob.glob(os.path.join(dir_path, "**"), recursive=True)
        relative_paths.sort()
        if len(file_extensions) > 0:
            relative_paths = [p for p in relative_paths if pathlib.Path(p).suffix in file_extensions]
        relative_paths = [os.path.relpath(f, dir_path) for f in relative_paths]
    else:
        raise FileNotFoundError(f"Specified path {dir_path} cannot be found or is not a directory.")
    return relative_paths

@contextlib.contextmanager
def pushd(new_dir: str):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)

class TypedItem(TypedDict):
    """
    We use a dictionary to store each item with resolved / inferred type information.
    """
    text: str
    start_line: NotRequired[int]
    start_col: NotRequired[int]
    end_line: NotRequired[int]
    end_col: NotRequired[int]
    syntax_role: NotRequired[str]
    type: str
    short_type: NotRequired[str]

def run_pytype(source_code_dir: str):
    rich.print("Running Pytype.")
    run_shell_command(['pytype', '-k', '-j', 'auto', '--no-report-errors', source_code_dir], ok_codes=(0,1))
    rich.print("Pytype done.")

def process_file(file: str, source_code_dir: str, pytype_options: Options) -> list[TypedItem]:
    file_path = os.path.join(source_code_dir, file)
    src = open(file_path, "r").read()
    src = textwrap.dedent(src.lstrip('\n'))
    module = annotate_ast.annotate_source(src, ast, pytype_options)
    annotations = [node for node in ast.walk(module) if hasattr(node, 'resolved_type')]

    list_items = []

    for node in annotations:
        if isinstance(node, ast.Name):
            new_item: TypedItem = {
                "text": node.id,
                "start_line": node.lineno,
                "start_col": node.col_offset,
                "syntax_role": "Variable",
                "type": str(node.resolved_type),
                "short_type": str(node.resolved_annotation)
            }
        elif isinstance(node, ast.Attribute):
            new_item: TypedItem = {
                "text": node.attr,
                "start_line": node.lineno,
                "start_col": node.col_offset,
                "syntax_role": "Attribute",
                "type": str(node.resolved_type),
                "short_type": str(node.resolved_annotation)
            }
        elif isinstance(node, ast.FunctionDef):
            new_item: TypedItem = {
                "text": node.name,
                "start_line": node.lineno,
                "start_col": node.col_offset,
                "syntax_role": "Function",
                "type": str(node.resolved_type),
                "short_type": str(node.resolved_annotation)
            }
        else:
            continue
        list_items.append(new_item)

    return list_items


def extract_types(source_code_dir: str, pytype_options: Options) -> dict[str, list[TypedItem]]:
    """
    Process source code, and extract the type info of variables/methods.
    Args:
        source_code_dir: the directory of source code.
    Returns:
        A dictionary storing extracted types per file, where key is a file path and
        value is a list of TypedItem, which itself is a dictionary (TypedDict).
    """
    # Get files to process
    relative_paths: list[str] = get_files_in_dir(source_code_dir, file_extensions=(".py",))
    responses: dict[str, list[TypedItem]] = {}
    with rich.progress.Progress() as progress:
        task = progress.add_task("Extracting the types", total=len(relative_paths))
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures: list[tuple[str, concurrent.futures.Future]] = []
            for file in relative_paths:
                f = executor.submit(process_file, file, source_code_dir, pytype_options)
                futures.append((file, f))
            for file, future in futures:
                try:
                    list_items = future.result(timeout=600)
                    responses[file] = list_items
                except Exception as e:
                    progress.print(f"failed to get types for {file}\nError: {e}\n")
                progress.advance(task)
            executor.shutdown(wait=False)
    return responses

def get_project_types(source_code_dir: str, output_file: str):
    with pushd(source_code_dir):
        run_pytype('.')
        results = extract_types('.', Options.create())
    with (open(output_file, "w") as fout):
        json.dump(results, fout, indent=2)


if __name__ == "__main__":
    get_project_types(sys.argv[1], sys.argv[2])
