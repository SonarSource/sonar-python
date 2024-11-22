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

from typing import Dict, Set, List

from serializer.symbols import ModuleSymbol, MergedFunctionSymbol, MergedClassSymbol, MergedOverloadedFunctionSymbol, \
    MergedModuleSymbol, MergedVarSymbol


def merge_modules(all_python_modules: Set[str], model_by_version: Dict[str, Dict[str, ModuleSymbol]]):
    merged_modules: Dict[str, MergedModuleSymbol] = {}
    for python_mod in all_python_modules:
        handled_classes: Dict[str, List[MergedClassSymbol]] = {}
        handled_funcs: Dict[str, List[MergedFunctionSymbol]] = {}
        handled_overloaded_functions: Dict[str, List[MergedOverloadedFunctionSymbol]] = {}
        handled_vars: Dict[str, List[MergedVarSymbol]] = {}
        merged_modules[python_mod] = MergedModuleSymbol(python_mod, handled_classes,
                                                        handled_funcs, handled_overloaded_functions, handled_vars)
        for version in model_by_version:
            model = model_by_version[version]
            # get current module
            if python_mod not in model:
                continue
            current_module = model[python_mod]
            merge_classes(current_module, handled_classes, version)
            merge_functions(current_module, handled_funcs, version)
            merge_overloaded_functions(current_module, handled_overloaded_functions, version)
            merge_vars(current_module, handled_vars, version)
    return merged_modules


def merge_vars(module_or_class, handled_vars, version):
    for var in module_or_class.vars:
        if var.fullname not in handled_vars:
            # doesn't exist: we add it
            handled_vars[var.fullname] = [MergedVarSymbol(var, [version])]
        else:
            compared = handled_vars[var.fullname]
            for elem in compared:
                if elem.var_symbol == var:
                    elem.valid_for.append(version)
                    break
            else:
                # no equivalent yet in the variations: add a new one
                handled_vars[var.fullname].append(MergedVarSymbol(var, [version]))


def merge_classes(current_module, handled_classes, version):
    for mod_class in current_module.classes:
        if mod_class.fullname not in handled_classes:
            functions = {}
            overloaded_functions = {}
            variables = {}
            merge_functions(mod_class, functions, version)
            merge_overloaded_functions(mod_class, overloaded_functions, version)
            merge_vars(mod_class, variables, version)
            handled_classes[mod_class.fullname] = [MergedClassSymbol(mod_class, functions,
                                                                     overloaded_functions, variables, [version])]
        else:
            # merge
            compared = handled_classes[mod_class.fullname]
            for elem in compared:
                if elem.class_symbol == mod_class:
                    functions = elem.methods
                    overloaded_functions = elem.overloaded_methods
                    variables = elem.vars
                    merge_functions(mod_class, functions, version)
                    merge_overloaded_functions(mod_class, overloaded_functions, version)
                    merge_vars(mod_class, variables, version)
                    elem.valid_for.append(version)
                    break
            else:
                functions = {}
                overloaded_functions = {}
                variables = {}
                merge_functions(mod_class, functions, version)
                merge_overloaded_functions(mod_class, overloaded_functions, version)
                merge_vars(mod_class, variables, version)
                compared.append(MergedClassSymbol(mod_class, functions, overloaded_functions, variables, [version]))


def merge_overloaded_functions(module_or_class, handled_overloaded_funcs, version):
    functions = (module_or_class.overloaded_functions
                 if isinstance(module_or_class, ModuleSymbol) else module_or_class.overloaded_methods)
    for func in functions:
        if func.fullname not in handled_overloaded_funcs:
            # doesn't exist: we add it
            handled_overloaded_funcs[func.fullname] = [MergedOverloadedFunctionSymbol(func, [version])]
        else:
            compared = handled_overloaded_funcs[func.fullname]
            for elem in compared:
                if elem.overloaded_function_symbol == func:
                    elem.valid_for.append(version)
                    break
            else:
                # no equivalent yet in the variations: add a new one
                handled_overloaded_funcs[func.fullname].append(MergedOverloadedFunctionSymbol(func, [version]))


def merge_functions(module_or_class, handled_funcs, version):
    functions = module_or_class.functions if isinstance(module_or_class, ModuleSymbol) else module_or_class.methods
    for func in functions:
        if func.fullname not in handled_funcs:
            # doesn't exist: we add it
            handled_funcs[func.fullname] = [MergedFunctionSymbol(func, [version])]
        else:
            compared = handled_funcs[func.fullname]
            for elem in compared:
                if elem.function_symbol == func:
                    elem.valid_for.append(version)
                    break
            else:
                # no equivalent yet in the variations: add a new one
                handled_funcs[func.fullname].append(MergedFunctionSymbol(func, [version]))
