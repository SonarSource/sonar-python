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
import pytype.pytd.pytd as pydt
from pytype.tools.annotate_ast import annotate_ast
from typing import TypedDict, Sequence, Dict
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
    type_details: Dict

    
def run_pytype(source_code_dir: str):
    rich.print("Running Pytype.")
    run_shell_command(['pytype', '-k', '-j', 'auto', '--no-report-errors', source_code_dir], ok_codes=(0,1))
    rich.print("Pytype done.")

def extract_node_info(node):
    new_item: TypedItem | None = None
    if isinstance(node, ast.AnnAssign) and hasattr(node.target, "resolved_type"):
        annotation_type = pydt.GenericType(pydt.ClassType("typing.Type"), [node.target.resolved_type])
        type_details = type_dict(annotation_type)
        new_item = {
            "text": node.annotation.id,
            "start_line": node.annotation.lineno,
            "start_col": node.annotation.col_offset,
            "syntax_role": "Annotation",
            "type": str(annotation_type),
            "short_type": str(node.target.resolved_type),
            "type_details": type_details
        }
    if hasattr(node, 'resolved_type'):
        resolved_type = node.resolved_type
        type_details = type_dict(resolved_type)
        if isinstance(node, ast.Name):
            new_item = {
                "text": node.id,
                "start_line": node.lineno,
                "start_col": node.col_offset,
                "syntax_role": "Variable",
                "type": str(resolved_type),
                "short_type": str(node.resolved_annotation),
                "type_details": type_details
            }
        elif isinstance(node, ast.Attribute):
            new_item= {
                "text": node.attr,
                "start_line": node.lineno,
                "start_col": node.col_offset,
                "syntax_role": "Attribute",
                "type": str(resolved_type),
                "short_type": str(node.resolved_annotation),
                "type_details": type_details
            }
        elif isinstance(node, ast.FunctionDef):
            new_item= {
                "text": node.name,
                "start_line": node.lineno,
                "start_col": node.col_offset,
                "syntax_role": "Function",
                "type": str(resolved_type),
                "short_type": str(node.resolved_annotation),
                "type_details": type_details
            }
    return new_item

def extract_method_type(inner_node, list_items, node):
    args = [ n.annotation.resolved_type for n in inner_node.args.args if hasattr(n.annotation, "resolved_type") ]
    class_type = pydt.ClassType(node.name)
    params = [class_type]
    params.extend(args)
    r = [n for n in inner_node.body if isinstance(n, ast.Return)]
    return_type = pydt.ClassType("builtins.NoneType")
    if len(r) == 1:
        return_type = r[0].value.resolved_type
    params.append(return_type)
    
    method_type = pydt.ClassType("typing.Callable")
    ret = pydt.CallableType(method_type, params)
    type_details = type_dict(ret)
    new_item= {
        "text": inner_node.name,
        "start_line": inner_node.lineno,
        "start_col": inner_node.col_offset,
        "syntax_role": "Method",
        "type": str(ret),
        "short_type": "",
        "type_details": type_details
    }
    list_items.append(new_item)

def extract_scope_info(scope_node, list_items):
    for node in ast.walk(scope_node):
        if isinstance(node, ast.ClassDef):
            for inner_node in node.body:
                if isinstance(inner_node, ast.FunctionDef):
                    extract_method_type(inner_node, list_items, node)
        else:
            t = extract_node_info( node)
            if t is not None:
                list_items.append(t)

    return list_items


def process_file(file: str, source_code_dir: str, pytype_options: Options) -> list[TypedItem]:
    file_path = os.path.join(source_code_dir, file)
    src = open(file_path, "r").read()
    src = textwrap.dedent(src.lstrip('\n'))
    module = annotate_ast.annotate_source(src, ast, pytype_options)
    list_items = []
    return extract_scope_info(module, list_items)

def type_dict(resolved_type) -> Dict:
    if isinstance(resolved_type, str) or isinstance(resolved_type, int) or isinstance(resolved_type, bool):
        return {
            "$class": "Primitive",
            "value": resolved_type
        }
    result = {
        "$class": type(resolved_type).__name__,
    }
    if resolved_type.name is not None:
        result["name"] = resolved_type.name
    if isinstance(resolved_type, pydt.GenericType):
        if resolved_type.base_type is not None:
            result["base_type"] = type_dict(resolved_type.base_type)
        if resolved_type.parameters is not None:
            parameters = list(map(lambda x: type_dict(x), resolved_type.parameters))
            result["parameters"] = parameters
    if isinstance(resolved_type, pydt._SetOfTypes):
        if resolved_type.type_list is not None:
            type_list = list(map(lambda x: type_dict(x), resolved_type.type_list))
            result["type_list"] = type_list
    if isinstance(resolved_type, pydt.Alias):
        if resolved_type.type is not None:
            result["type"] = type_dict(resolved_type.type)
    if isinstance(resolved_type, pydt.Module):
        if resolved_type.module_name is not None:
            result["module_name"] = resolved_type.module_name
    if isinstance(resolved_type, pydt.Literal):
        result["value"] = type_dict(resolved_type.value)
    if isinstance(resolved_type, pydt.TypeParameter):
        if resolved_type.constraints is not None:
            result["constraints"] = list(map(lambda x: type_dict(x), resolved_type.constraints))
        if resolved_type.bound is not None:
            result["bound"] = type_dict(resolved_type.bound)
        if resolved_type.scope is not None:
            result["scope"] = resolved_type.scope
    return result

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
        rich.print("Dumping data into the json")
        json.dump(results, fout, indent=2)
        rich.print("Dumping done")


if __name__ == "__main__":
    # get_project_types("sample", "types.json")
    get_project_types(sys.argv[1], sys.argv[2])
