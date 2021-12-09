#
# SonarQube Python Plugin
# Copyright (C) 2011-2021 SonarSource SA
# mailto:info AT sonarsource DOT com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

import os

from mypy import build, options

from serializer import symbols_merger, symbols

STDLIB_PATH = "../resources/typeshed/stdlib"
CURRENT_PATH = os.path.dirname(__file__)


def get_options(python_version=(3, 8)):
    opt = options.Options()
    # Setting incremental to false to avoid issues with mypy caching
    opt.incremental = False
    opt.python_version = python_version
    return opt


def build_single_module(module_fqn: str, category="stdlib", python_version=(3, 8)):
    opt = get_options(python_version)
    module_source = load_single_module(module_fqn, category)
    build_result = build.build([module_source], opt)
    built_file = build_result.files.get(module_fqn)
    return built_file


def load_single_module(module_fqn: str, category="stdlib"):
    module_path = module_fqn
    if '.' in module_fqn:
        module_path = module_fqn.replace('.', "/")
    if os.path.isfile(path := os.path.join(CURRENT_PATH,
                                           f"../resources/typeshed/{category}/{module_path}.pyi")):
        module_source = build.BuildSource(path, module_fqn)
    elif os.path.isfile(path := os.path.join(CURRENT_PATH,
                                             f"../resources/typeshed/{category}/{module_path}/__init__.pyi")):
        module_source = build.BuildSource(path, module_fqn)
    else:
        raise FileNotFoundError(f"No stub found for module {module_fqn}")
    return module_source


def walk_typeshed_stdlib(opt: options.Options = get_options()):
    source_list = []
    generate_python2_stdlib = opt.python_version < (3, 0)
    relative_path = STDLIB_PATH if not generate_python2_stdlib else f"{STDLIB_PATH}/@python2"
    path = os.path.join(CURRENT_PATH, relative_path)
    for root, dirs, files in os.walk(path):
        package_name = root.replace(path, "").replace("\\", ".").replace("/", ".").lstrip(".")
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


def serialize_typeshed_stdlib(output_dir_name="output", python_version=(3, 8), is_debug=False):
    """ Serialize semantic model for Python standard library
    :param output_dir_name: Optional output directory name
    :param python_version: Optional version of Python to use for serialization
    :param is_debug: debug flag
    """
    output_dir_name = output_dir_name if python_version >= (3, 0) else f"{output_dir_name}@python2"
    opt = get_options(python_version)
    build_result = walk_typeshed_stdlib(opt)
    for file in build_result.files:
        module_symbol = symbols.ModuleSymbol(build_result.files.get(file))
        symbols.save_module(module_symbol, is_debug=is_debug, debug_dir=output_dir_name)


def serialize_typeshed_stdlib_multiple_python_version():
    """ Serialize semantic model for Python stdlib versions from 3.5 to 3.9
    """
    for minor in range(5, 10):
        serialize_typeshed_stdlib(f"output3{minor}", (3, minor), is_debug=True)


def save_merged_symbols(is_debug=False):
    merged_modules = symbols_merger.merge_multiple_python_versions()
    for mod in merged_modules:
        symbols.save_module(merged_modules[mod], is_debug=is_debug, debug_dir="output_merge")


def main():
    save_merged_symbols()


if __name__ == '__main__':
    main()
