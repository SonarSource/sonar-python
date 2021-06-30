import os
import sys
import collections

from mypy import build, options

from serializer.symbols import save_module

VersionInfoTuple = collections.namedtuple('version_info', ['major', 'minor', 'micro', 'releaselevel', 'serial'])
STDLIB_PATH = "../resources/typeshed/stdlib"


def get_options(python_version=None):
    opt = options.Options()
    # Setting incremental to false to avoid issues with mypy caching
    opt.incremental = False
    if python_version is not None:
        opt.python_version = python_version
    return opt


def build_single_module(module_fqn: str, category="stdlib", opt=get_options()):
    module_source = load_single_module(module_fqn, category)
    build_result = build.build([module_source], opt)
    built_file = build_result.files.get(module_fqn)
    return built_file


def load_single_module(module_fqn: str, category="stdlib"):
    module_path = module_fqn
    if '.' in module_fqn:
        module_path = module_fqn.replace('.', "/")
    if os.path.isfile(path := f"../resources/typeshed/{category}/{module_path}.pyi"):
        module_source = build.BuildSource(path, module_fqn)
    elif os.path.isfile(path := f"../resources/typeshed/{category}/{module_path}/__init__.pyi"):
        module_source = build.BuildSource(path, module_fqn)
    else:
        raise FileNotFoundError(f"No stub found for module {module_fqn}")
    return module_source


def walk_typeshed_stdlib(opt=get_options()):
    source_list = []
    generate_python2_stdlib = opt.python_version < (3, 0)
    path = STDLIB_PATH if not generate_python2_stdlib else f"{STDLIB_PATH}/@python2"
    for root, dirs, files in os.walk(path):
        package_name = root.replace(path, "").replace("\\", ".").lstrip(".")
        if not generate_python2_stdlib and "python2" in package_name:
            # Avoid python2 stubs
            continue
        for file in files:
            if not file.endswith(".pyi"):
                # Only consider actual stubs
                continue
            module_name = file.replace(".pyi", "")
            fq_module_name = f"{package_name}.{module_name}" if package_name != "" else module_name
            if module_name == "__init__":
                fq_module_name = package_name
            file_path = f"{root}/{file}"
            source = build.BuildSource(file_path, module=fq_module_name)
            source_list.append(source)
    build_result = build.build(source_list, opt)
    return build_result


def serialize_typeshed_stdlib(output_dir_name="output", python_version=(3, 8)):
    """ Serialize semantic model for Python standard library
    :param output_dir_name: Optional output directory name
    :param python_version: Optional version of Python to use for serialization
    """
    output_dir_name = output_dir_name if python_version >= (3, 0) else f"{output_dir_name}@python2"
    opt = get_options(python_version)
    build_result = walk_typeshed_stdlib(opt)
    for file in build_result.files:
        save_module(build_result.files.get(file), output_dir_name=output_dir_name)


def serialize_typeshed_stdlib_multiple_python_version():
    """ Serialize semantic model for Python stdlib versions from 3.5 to 3.9
    """
    for minor in range(5, 10):
        sys.version_info = VersionInfoTuple(3, minor, 0, 'final', 0)
        serialize_typeshed_stdlib(f"output3{minor}", (3, minor))


if __name__ == '__main__':
    #annoy_mypy_file = build_single_module("annoy", "stubs/annoy")
    #save_module(annoy_mypy_file, output_dir_name="annoy_output")
    serialize_typeshed_stdlib()
