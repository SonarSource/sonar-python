#
# SonarQube Python Plugin
# Copyright (C) 2011-2023 SonarSource SA
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

from mypy import build, options, nodes

from serializer import symbols_merger, symbols

STDLIB_PATH = "../resources/typeshed/stdlib"
STDLIB_INTERNAL_PATH = "../resources/typeshed-internal/stdlib"
STUBS_PATH = "../resources/typeshed-internal/stubs"
CURRENT_PATH = os.path.dirname(__file__)
THIRD_PARTIES_STUBS = os.listdir(os.path.join(CURRENT_PATH, STUBS_PATH))
CUSTOM_STUBS_PATH = "../resources/custom"
STUBGEN_GENERATED_PATH = "../resources/stubgen_generated"
SONAR_CUSTOM_BASE_STUB_MODULE = "SonarPythonAnalyzerFakeStub"


def get_options(python_version=(3, 8)):
    opt = options.Options()
    # Setting incremental to false to avoid issues with mypy caching
    opt.incremental = False
    opt.python_version = python_version
    opt.platform = "linux"
    return opt


def walk_typeshed_stdlib(opt: options.Options = get_options()):
    generate_python2_stdlib = opt.python_version < (3, 0)
    relative_path = STDLIB_PATH if not generate_python2_stdlib else f"{STDLIB_INTERNAL_PATH}/@python2"
    source_list, source_paths = get_sources(relative_path, generate_python2_stdlib)
    build_result = build.build(source_list, opt)
    return build_result, source_paths


def walk_typeshed_third_parties(opt: options.Options = get_options()):
    source_list = []
    source_paths = set()
    generate_python2 = opt.python_version < (3, 0)
    for third_party_stub in THIRD_PARTIES_STUBS:
        stub_path = os.path.join(STUBS_PATH, third_party_stub)
        relative_path = stub_path if not generate_python2 else f"{stub_path}/@python2"
        src_list, src_paths = get_sources(relative_path, generate_python2)
        source_list.extend(src_list)
        source_paths = source_paths.union(src_paths)
    build_result = build.build(source_list, opt)
    return build_result, source_paths


def walk_custom_stubs(opt: options.Options = get_options(), path=CUSTOM_STUBS_PATH):
    source_list, source_paths = get_sources(path, False)
    build_result = build.build(source_list, opt)
    return build_result, source_paths


def get_sources(relative_path: str, generate_python2: bool, extension=".pyi"):
    source_list = []
    source_paths = set()
    path = os.path.join(CURRENT_PATH, relative_path)
    for root, dirs, files in os.walk(path):
        package_name = root.replace(path, "").replace("\\", ".").replace("/", ".").lstrip(".")
        if not generate_python2 and "python2" in package_name:
            # Avoid python2 stubs
            continue
        for file in files:
            if not file.endswith(extension):
                # Only consider actual stubs
                continue
            module_name = file.replace(extension, "")
            fq_module_name = f"{package_name}.{module_name}" if package_name != "" else module_name
            if module_name == "__init__":
                fq_module_name = package_name
            file_path = f"{root}/{file}"
            source = build.BuildSource(file_path, module=fq_module_name)
            source_list.append(source)
            source_paths.add(source.path)
    return source_list, source_paths


def serialize_typeshed_stdlib(output_dir_name="output", python_version=(3, 8), is_debug=False):
    """ Serialize semantic model for Python standard library
    :param output_dir_name: Optional output directory name
    :param python_version: Optional version of Python to use for serialization
    :param is_debug: debug flag
    """
    output_dir_name = output_dir_name if python_version >= (3, 0) else f"{output_dir_name}@python2"
    opt = get_options(python_version)
    build_result, _ = walk_typeshed_stdlib(opt)
    for file in build_result.files:
        module_symbol = symbols.ModuleSymbol(build_result.files.get(file))
        symbols.save_module(module_symbol, "stdlib_protobuf", is_debug=is_debug, debug_dir=output_dir_name)


def serialize_custom_stubs(output_dir_name="output", python_version=(3, 8), is_debug=False):
    path = os.path.join(CURRENT_PATH, CUSTOM_STUBS_PATH)
    opt = get_options(python_version)
    build_result, _ = walk_custom_stubs(opt)
    for file in build_result.files:
        if file == SONAR_CUSTOM_BASE_STUB_MODULE:
            continue
        current_file = build_result.files.get(file)
        if not current_file.path.startswith(path):
            continue
        module_symbol = symbols.ModuleSymbol(current_file)
        symbols.save_module(module_symbol, "custom_protobuf", is_debug=is_debug, debug_dir=output_dir_name)


def serialize_stubgen_generated(output_dir_name="output_stubgen_generated", python_version=(3, 8), is_debug=False):
    path = os.path.join(CURRENT_PATH, STUBGEN_GENERATED_PATH)
    opt = get_options(python_version)
    build_result, _ = walk_custom_stubs(opt, path=STUBGEN_GENERATED_PATH)
    for file in build_result.files:
        if file == SONAR_CUSTOM_BASE_STUB_MODULE:
            continue
        current_file = build_result.files.get(file)
        if not current_file.path.startswith(path):
            continue
        module_symbol = symbols.ModuleSymbol(current_file)
        symbols.save_module(module_symbol, "stubgen_protobuf", is_debug=is_debug, debug_dir=output_dir_name)


def serialize_flask(output_dir_name="output", python_version=(3, 10), is_debug=True):
    path = os.path.join(CURRENT_PATH, "../my_cache")
    opt = get_options(python_version)
    opt.export_types = True
    source = build.BuildSource(path, module="flask_test.test")
    source_list, source_paths = get_sources(path, False, extension=".data.json")
    import json
    for path in source_paths:
        with open(path) as json_file:
            current_file = json.load(json_file)
            mypy_file = nodes.SymbolNode.deserialize(current_file)
            module_symbol = symbols.ModuleSymbol(mypy_file)
            symbols.save_module(module_symbol, "flask_poc", is_debug=is_debug, debug_dir=output_dir_name)



def serialize_typeshed_stdlib_multiple_python_version():
    """ Serialize semantic model for Python stdlib versions from 3.5 to 3.9
    """
    for minor in range(5, 10):
        serialize_typeshed_stdlib(f"output3{minor}", (3, minor), is_debug=True)


def save_merged_symbols(is_debug=False, is_third_parties=False):
    dir_name = "third_party_protobuf" if is_third_parties else "stdlib_protobuf"
    merged_modules = symbols_merger.merge_multiple_python_versions(is_third_parties)
    for mod in merged_modules:
        symbols.save_module(merged_modules[mod], dir_name, is_debug=is_debug, debug_dir="output_merge")


def main():
#    save_merged_symbols(is_debug=True)
#    save_merged_symbols(is_third_parties=True, is_debug=True)
    # serialize_custom_stubs()
#    serialize_typeshed_stdlib(is_debug=True)
    # serialize_stubgen_generated(is_debug=True)
    serialize_flask()


if __name__ == '__main__':
    main()
