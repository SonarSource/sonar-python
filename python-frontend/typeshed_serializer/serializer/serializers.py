#
# SonarQube Python Plugin
# Copyright (C) 2011-2024 SonarSource SA
# mailto:info AT sonarsource DOT com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Sonar Source-Available License for more details.
#
# You should have received a copy of the Sonar Source-Available License
# along with this program; if not, see https://sonarsource.com/license/ssal/
#

import os
import logging
import sys
from abc import abstractmethod, ABC


from mypy import options, build

from serializer import symbols
from serializer.symbols import ModuleSymbol
from serializer.symbols_merger import merge_modules
from utils.folder_manager import FolderManager

STDLIB_PATH = "../resources/typeshed/stdlib"
STUBS_PATH = "../resources/typeshed/stubs"
SKLEARN_STUBS_PATH = "../resources/python-type-stubs/stubs"
CURRENT_PATH = os.path.dirname(__file__)
THIRD_PARTIES_STUBS = os.listdir(os.path.join(CURRENT_PATH, STUBS_PATH))
CUSTOM_STUBS_PATH = "../resources/custom"
SONAR_CUSTOM_BASE_STUB_MODULE = "SonarPythonAnalyzerFakeStub"
IMPORTER_FILE_NAME = "../resources/importer/sonar_third_party_libs.py"
IMPORTER_FQN = "sonar_third_party_libs"
SUPPORTED_PYTHON_VERSIONS = ((3, 6), (3, 7), (3, 8), (3, 9), (3, 10), (3, 11))


def get_options(python_version=(3, 8)):
    opt = options.Options()
    # Setting incremental to false to avoid issues with mypy caching
    opt.incremental = False
    opt.python_version = python_version
    opt.platform = "linux"
    return opt


def walk_typeshed_stdlib(opt: options.Options = get_options()):
    source_list, source_paths = get_sources(STDLIB_PATH)
    build_result = build.build(source_list, opt)
    return build_result, source_paths


def walk_typeshed_third_parties(opt: options.Options = get_options()):
    source_list = []
    source_paths = set()
    for third_party_stub in THIRD_PARTIES_STUBS:
        stub_path = os.path.join(STUBS_PATH, third_party_stub)
        src_list, src_paths = get_sources(stub_path)
        source_list.extend(src_list)
        source_paths = source_paths.union(src_paths)
    build_result = build.build(source_list, opt)
    return build_result, source_paths


def get_sources(relative_path: str):
    source_list = []
    source_paths = set()
    path = os.path.join(CURRENT_PATH, relative_path)
    for root, dirs, files in os.walk(path):
        package_name = (
            root.replace(path, "").replace("\\", ".").replace("/", ".").lstrip(".")
        )
        if "python2" in package_name:
            # Avoid python2 stubs
            continue
        for file in files:
            if not file.endswith(".pyi"):
                # Only consider actual stubs
                continue
            module_name = file.replace(".pyi", "")
            fq_module_name = (
                f"{package_name}.{module_name}" if package_name != "" else module_name
            )
            if module_name == "__init__":
                fq_module_name = package_name
            file_path = f"{root}/{file}"
            source = build.BuildSource(file_path, module=fq_module_name)
            source_list.append(source)
            source_paths.add(source.path)
    return source_list, source_paths


class Serializer(ABC):
    save_location: str
    output_folder: str

    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s [%(levelname)s] --- %(message)s ---"
    )
    logger.setLevel(logging.INFO)
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    def __init__(self, is_debug=False, python_version=(3, 8)):
        self.is_debug = is_debug
        self.opt = get_options(python_version)

    def serialize(self, output_dir_name="output") -> None:
        build_result, source_paths = self.get_build_result(self.opt)
        for file in build_result.files:
            if self.is_exception(file, build_result, source_paths):
                continue
            current_file = build_result.files.get(file)
            module_symbol = symbols.ModuleSymbol(current_file)
            symbols.save_module(
                module_symbol,
                self.output_folder,
                is_debug=self.is_debug,
                debug_dir=output_dir_name,
            )

    @abstractmethod
    def get_build_result(self, opt=get_options()) -> tuple[build.BuildResult, set[str]]:
        """returns a tuple containing the semantic model of the project and the paths of its sources"""

    @abstractmethod
    def is_exception(self, file, build_result, source_paths) -> bool:
        """returns True if the given file should be skipped from serialization"""


class TypeshedSerializer(Serializer):
    EXCLUDED_PACKAGES = ["win32"]

    def __init__(self, is_third_parties=False, is_debug=False):
        super().__init__(is_debug)
        self.is_third_parties = is_third_parties
        self.save_location = (
            "third_party_protobuf" if is_third_parties else "stdlib_protobuf"
        )
        self.output_folder = f"{FolderManager.output_folder}/{self.save_location}"

    def serialize_merged_modules(self):
        merged_modules = self.get_merged_modules()
        for mod in merged_modules:
            symbols.save_module(
                merged_modules[mod],
                self.output_folder,
                is_debug=self.is_debug,
                debug_dir="output_merge",
            )

    def get_merged_modules(self):
        model_by_version = self.build_for_every_python_version()
        all_python_modules: set[str] = set()
        for version in model_by_version:
            model = model_by_version[version]
            for module_fqn in model:
                mod: ModuleSymbol = model[module_fqn]
                all_python_modules.add(mod.fullname)
        merged_modules = merge_modules(all_python_modules, model_by_version)
        return merged_modules

    def build_for_every_python_version(self):
        model_by_version: dict[str, dict[str, ModuleSymbol]] = {}
        for major, minor in SUPPORTED_PYTHON_VERSIONS:
            opt = get_options((major, minor))
            self.logger.info(f"Building for python version {major}.{minor}")
            build_result, source_paths = self.get_build_result(opt=opt)
            modules = {}
            for file in build_result.files:
                path = build_result.files[file].path
                if self.is_third_parties and (
                    path not in source_paths
                    or any(
                        excluded_package in path
                        for excluded_package in self.EXCLUDED_PACKAGES
                    )
                ):
                    # build_result contains more modules from stdlib unrelated to third_parties
                    continue

                ms = ModuleSymbol(build_result.files.get(file))
                modules[ms.fullname] = ms
            model_by_version[f"{major}{minor}"] = modules
        return model_by_version

    def get_build_result(self, opt=get_options()):
        build_result, source_paths = (
            walk_typeshed_third_parties(opt)
            if self.is_third_parties
            else walk_typeshed_stdlib(opt)
        )
        return build_result, source_paths

    def is_exception(self, file, build_result, source_paths):
        file_path = build_result.files[file].path
        return self.is_third_parties and file_path not in source_paths


class ImporterSerializer(Serializer):
    save_location = "third_party_protobuf_mypy"
    output_folder = f"{FolderManager.output_folder}/{save_location}"

    def get_build_result(self, opt=get_options()):
        path = os.path.join(CURRENT_PATH, IMPORTER_FILE_NAME)
        source = build.BuildSource(path, module=IMPORTER_FQN)
        build_result = build.build([source], options=self.opt)
        return build_result, {path}

    def is_exception(self, file, build_result, source_paths):
        # SONARPY-1499: Numpy information is transitively gathered from Pandas (SONARPY-1487)
        # It is currently filtered out to avoid the risk of false positives
        return file == "sonar_third_party_libs" or file.startswith("numpy")


class MicrosoftStubsSerializer(Serializer):
    save_location = "third_party_protobuf"
    output_folder = f"{FolderManager.output_folder}/{save_location}"

    def get_build_result(self, opt=get_options()):
        src_list, src_paths = get_sources(SKLEARN_STUBS_PATH)
        build_result = build.build(src_list, opt)
        return build_result, src_paths

    def is_exception(self, file, build_result, source_paths):
        file_path = build_result.files[file].path
        # Filtering out KDTree and BallTree to avoid FPs
        return (
            "sklearn" not in file_path
            or "_kd_tree" in file_path
            or "_ball_tree" in file_path
        )


class CustomStubsSerializer(Serializer):
    path = os.path.join(CURRENT_PATH, CUSTOM_STUBS_PATH)
    save_location = "custom_protobuf"
    output_folder = f"{FolderManager.output_folder}/{save_location}"

    def get_build_result(self, opt=get_options()):
        source_list, source_paths = get_sources(CUSTOM_STUBS_PATH)
        build_result = build.build(source_list, self.opt)
        return build_result, source_paths

    def is_exception(self, file, build_result, source_paths=None):
        return file == SONAR_CUSTOM_BASE_STUB_MODULE or not build_result.files.get(
            file
        ).path.startswith(self.path)
