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

from unittest.mock import Mock

from serializer import symbols, symbols_merger, serializers
from serializer.serializers import TypeshedSerializer
from serializer.symbols import MergedModuleSymbol, TypeKind


def test_build_multiple_python_version(typeshed_stdlib):
    serializers.walk_typeshed_stdlib = Mock(return_value=(typeshed_stdlib, set()))
    model_by_version = TypeshedSerializer().build_for_every_python_version()
    assert set(model_by_version.keys()) == {'38', '39', '310', '311', '312', '313'}


def test_merge_multiple_python_versions(typeshed_stdlib):
    serializers.walk_typeshed_stdlib = Mock(return_value=(typeshed_stdlib, set()))
    merged_modules = TypeshedSerializer().get_merged_modules()
    for mod in merged_modules.values():
        assert isinstance(mod, MergedModuleSymbol)
    assert len(merged_modules) == 45


def test_basic_module_merge(typeshed_stdlib):
    abc_module_symbol = symbols.ModuleSymbol(typeshed_stdlib.files.get("abc"))

    # Merge single module
    merged_modules = symbols_merger.merge_modules({"abc"}, {"37": {"abc": abc_module_symbol}})
    assert_abc_merged_module(merged_modules, ["37"])

    # Merge identical modules
    merged_modules = symbols_merger.merge_modules({"abc"}, {"37": {"abc": abc_module_symbol},
                                                            "38": {"abc": abc_module_symbol}})
    assert_abc_merged_module(merged_modules, ["37", "38"])

    # Merged module symbol conversion to proto
    abc_merged_symbol = merged_modules["abc"]
    abc_merged_symbol_proto = abc_merged_symbol.to_proto()
    assert abc_merged_symbol_proto.fully_qualified_name == abc_merged_symbol.fullname
    assert_merged_class_symbol_to_proto(abc_merged_symbol_proto.classes, abc_merged_symbol.classes)
    assert_merged_function_symbol_to_proto(abc_merged_symbol_proto.functions, abc_merged_symbol.functions)


def test_merged_symbols_to_proto(typeshed_stdlib):
    ssl_module_symbol = symbols.ModuleSymbol(typeshed_stdlib.files.get("ssl"))

    # Merge single module
    merged_modules = symbols_merger.merge_modules({"ssl"}, {"37": {"ssl": ssl_module_symbol}})
    assert len(merged_modules) == 1
    ssl_merged_module_symbol = merged_modules["ssl"]

    # Merged module symbol conversion to proto
    ssl_merged_symbol_proto = ssl_merged_module_symbol.to_proto()
    assert ssl_merged_symbol_proto.fully_qualified_name == ssl_merged_module_symbol.fullname

    assert_merged_class_symbol_to_proto(ssl_merged_symbol_proto.classes, ssl_merged_module_symbol.classes)
    assert_merged_function_symbol_to_proto(ssl_merged_symbol_proto.functions, ssl_merged_module_symbol.functions)


def test_basic_functions_merge(typeshed_stdlib):
    abc_module_symbol = symbols.ModuleSymbol(typeshed_stdlib.files.get("abc"))
    handled_funcs = {}

    # merge from empty
    symbols_merger.merge_functions(abc_module_symbol, handled_funcs, '37')
    assert len(handled_funcs) == len(abc_module_symbol.functions)
    for merged_func in handled_funcs.values():
        assert len(merged_func) == 1
        assert merged_func[0].valid_for == ['37']

    # merge with identical model
    symbols_merger.merge_functions(abc_module_symbol, handled_funcs, '38')
    assert len(handled_funcs) == len(abc_module_symbol.functions)
    for merged_func in handled_funcs.values():
        assert len(merged_func) == 1
        assert merged_func[0].valid_for == ['37', '38']


def test_basic_classes_merge(typeshed_stdlib):
    abc_module_symbol = symbols.ModuleSymbol(typeshed_stdlib.files.get("abc"))
    handled_classes = {}

    # merge from empty
    symbols_merger.merge_classes(abc_module_symbol, handled_classes, '37')
    assert len(handled_classes) == len(abc_module_symbol.classes)
    for merged_class in handled_classes.values():
        assert len(merged_class) == 1
        assert merged_class[0].valid_for == ['37']

    # merge with identical model
    symbols_merger.merge_classes(abc_module_symbol, handled_classes, '38')
    assert len(handled_classes) == len(abc_module_symbol.classes)
    for merged_class in handled_classes.values():
        assert len(merged_class) == 1
        assert merged_class[0].valid_for == ['37', '38']


def test_overloaded_functions_merge(typeshed_stdlib):
    ssl_module_symbol = symbols.ModuleSymbol(typeshed_stdlib.files.get("ssl"))
    handled_overloaded_funcs = {}
    socket_class_symbol = [c for c in ssl_module_symbol.classes if c.fullname == 'ssl.SSLSocket'][0]

    # merge from empty
    symbols_merger.merge_overloaded_functions(socket_class_symbol, handled_overloaded_funcs, '37')
    assert len(handled_overloaded_funcs) == len(socket_class_symbol.overloaded_methods)
    for merged_overloaded_func in handled_overloaded_funcs.values():
        assert len(merged_overloaded_func) == 1
        assert merged_overloaded_func[0].valid_for == ['37']

    # merge with identical model
    symbols_merger.merge_overloaded_functions(socket_class_symbol, handled_overloaded_funcs, '38')
    for merged_overloaded_func in handled_overloaded_funcs.values():
        assert len(merged_overloaded_func) == 1
        assert merged_overloaded_func[0].valid_for == ['37', '38']


def test_actual_module_merge(fake_module_36_38):
    fake_module_36 = symbols.ModuleSymbol(fake_module_36_38[0])
    fake_module_38 = symbols.ModuleSymbol(fake_module_36_38[1])
    merged_modules = symbols_merger.merge_modules({"fakemodule"}, {"36": {"fakemodule": fake_module_36},
                                                                   "38": {"fakemodule": fake_module_38}})
    merged_fakemodule_module = merged_modules['fakemodule']
    classes_dict = merged_fakemodule_module.classes

    assert len(classes_dict) == 5

    # Class unique to Python 3.6 is present
    fakemodule_someclassunique36_symbols = classes_dict['fakemodule.SomeClassUnique36']
    assert len(fakemodule_someclassunique36_symbols) == 1
    merged_someclassunique36_symbol = fakemodule_someclassunique36_symbols[0]
    assert merged_someclassunique36_symbol.valid_for == ["36"]

    # Class unique to Python 3.8 is present
    fakemodule_someclassunique36_symbols = classes_dict['fakemodule.SomeClassUnique38']
    assert len(fakemodule_someclassunique36_symbols) == 1
    merged_someclassunique36_symbol = fakemodule_someclassunique36_symbols[0]
    assert merged_someclassunique36_symbol.valid_for == ["38"]

    # Class common to Python 3.6 and Python 3.8 is present
    fakemodule_commonclass_symbols = classes_dict['fakemodule.CommonClass']
    assert len(fakemodule_commonclass_symbols) == 1
    commonclass_symbol = fakemodule_commonclass_symbols[0]
    # Some methods are common
    assert commonclass_symbol.methods['fakemodule.CommonClass.common_method'][0].valid_for == ["36", "38"]
    # Some methods exist only in a given Python version
    assert commonclass_symbol.methods['fakemodule.CommonClass.method_unique_36'][0].valid_for == ["36"]
    assert commonclass_symbol.methods['fakemodule.CommonClass.method_unique_38'][0].valid_for == ["38"]
    # Some methods have different definitions depending on the Python version
    assert commonclass_symbol.methods['fakemodule.CommonClass.common_method_multiple_definition'][0].valid_for == ["36"]
    assert commonclass_symbol.methods['fakemodule.CommonClass.common_method_multiple_definition'][1].valid_for == ["38"]

    functions_dict = merged_fakemodule_module.functions
    assert len(functions_dict) == 5

    common_function_symbols = functions_dict['fakemodule.common_function']
    assert len(common_function_symbols) == 1
    assert common_function_symbols[0].valid_for == ["36", "38"]

    common_function_wildcard_imported = functions_dict['fakemodule_imported.common_imported_func']
    assert len(common_function_wildcard_imported) == 1
    assert common_function_wildcard_imported[0].valid_for == ["36", "38"]

    assert 'fakemodule_imported._private_func' not in functions_dict

    function_unique_36 = functions_dict['fakemodule.function_unique_36']
    assert len(function_unique_36) == 1
    assert function_unique_36[0].valid_for == ["36"]

    function_unique_38 = functions_dict['fakemodule.function_unique_38']
    assert len(function_unique_38) == 1
    assert function_unique_38[0].valid_for == ["38"]

    common_function_multiple_defs = functions_dict['fakemodule.common_function_multiple_defs']
    assert len(common_function_multiple_defs) == 2
    assert common_function_multiple_defs[0].valid_for == ["36"]
    assert common_function_multiple_defs[1].valid_for == ["38"]

    overloaded_functions_dict = merged_fakemodule_module.overloaded_functions
    assert len(overloaded_functions_dict) == 4
    common_overloaded_function_symbols = overloaded_functions_dict['fakemodule.common_overloaded_function']
    assert len(common_overloaded_function_symbols) == 1
    assert common_overloaded_function_symbols[0].valid_for == ["36", "38"]

    overloaded_function_unique_36 = overloaded_functions_dict['fakemodule.overloaded_function_36']
    assert len(overloaded_function_unique_36) == 1
    assert overloaded_function_unique_36[0].valid_for == ["36"]

    overloaded_function_unique_38 = overloaded_functions_dict['fakemodule.overloaded_function_38']
    assert len(overloaded_function_unique_38) == 1
    assert overloaded_function_unique_38[0].valid_for == ["38"]

    fakemodule_proto = merged_fakemodule_module.to_proto()
    flattened_overloaded_funcs = \
        [func for alternatives in merged_fakemodule_module.overloaded_functions.values() for func in alternatives]
    assert len(fakemodule_proto.overloaded_functions) == len(flattened_overloaded_funcs)

    all_vars = merged_fakemodule_module.vars
    assert len(all_vars) == 8
    common_var = all_vars['fakemodule.common_var']
    assert len(common_var) == 1
    assert common_var[0].valid_for == ["36", "38"]
    var_symbol = common_var[0].var_symbol
    assert var_symbol.name == "common_var"
    assert var_symbol.fullname == "fakemodule.common_var"
    assert var_symbol.type.fully_qualified_name == "builtins.bool"
    assert var_symbol.is_imported_module is False

    unique_var_36 = all_vars['fakemodule.unique_var_36']
    assert len(unique_var_36) == 1
    assert unique_var_36[0].valid_for == ["36"]

    unique_var_38 = all_vars['fakemodule.unique_var_38']
    assert len(unique_var_38) == 1
    assert unique_var_38[0].valid_for == ["38"]

    var_multiple_defs = all_vars['fakemodule.var_multiple_defs']
    assert len(var_multiple_defs) == 2
    definition_36 = var_multiple_defs[0]
    definition_38 = var_multiple_defs[1]
    assert definition_36.valid_for == ["36"]
    assert definition_36.var_symbol.type.fully_qualified_name == "builtins.int"
    assert definition_38.valid_for == ["38"]
    assert definition_38.var_symbol.type.fully_qualified_name == "builtins.str"

    alias = all_vars['fakemodule.alias']
    assert len(alias) == 1
    alias_symbol = alias[0].var_symbol
    assert alias_symbol.type.fully_qualified_name == "builtins.function"
    assert alias_symbol.type.kind == TypeKind.CALLABLE
    assert alias_symbol.type.pretty_printed_name == "CallableType[builtins.function]"

    imported_math = all_vars['math']
    assert len(imported_math) == 1
    assert imported_math[0].var_symbol.is_imported_module is True

    imported_sys = all_vars['sys.flags']
    assert len(imported_sys) == 1
    assert imported_sys[0].var_symbol.name == "my_flags"

    fakemodule_class_with_fields_symbols = classes_dict['fakemodule.ClassWithFields']
    assert len(fakemodule_class_with_fields_symbols) == 1
    fakemodule_class_symbol = fakemodule_class_with_fields_symbols[0]
    # Some fields are common
    assert fakemodule_class_symbol.vars['fakemodule.ClassWithFields.common_field'][0].valid_for == ["36", "38"]
    # Some fields exist only in a given Python version
    assert fakemodule_class_symbol.vars['fakemodule.ClassWithFields.field_unique_36'][0].valid_for == ["36"]
    assert fakemodule_class_symbol.vars['fakemodule.ClassWithFields.field_unique_38'][0].valid_for == ["38"]
    # Some fields have different definitions depending on the Python version
    assert fakemodule_class_symbol.vars['fakemodule.ClassWithFields.field_multiple_defs'][0].valid_for == ["36"]
    assert fakemodule_class_symbol.vars['fakemodule.ClassWithFields.field_multiple_defs'][1].valid_for == ["38"]


def assert_merged_class_symbol_to_proto(merged_classes_proto, merged_classes):
    assert len(merged_classes_proto) == len(merged_classes)
    for merged_class_proto in merged_classes_proto:
        merged_class_symbol = merged_classes[merged_class_proto.fully_qualified_name]
        assert len(merged_class_symbol) == 1
        original_class_symbol = merged_class_symbol[0]
        assert merged_class_proto.has_decorators == original_class_symbol.class_symbol.has_decorators
        assert merged_class_proto.has_metaclass == original_class_symbol.class_symbol.has_metaclass
        assert merged_class_proto.is_enum == original_class_symbol.class_symbol.is_enum
        assert merged_class_proto.is_generic == original_class_symbol.class_symbol.is_generic
        assert merged_class_proto.is_protocol == original_class_symbol.class_symbol.is_protocol
        assert len(merged_class_proto.methods) == len(original_class_symbol.methods)
        if len(merged_class_proto.methods) > 0:
            assert_merged_function_symbol_to_proto(merged_class_proto.methods, original_class_symbol.methods)
        assert len(merged_class_proto.overloaded_methods) == len(original_class_symbol.overloaded_methods)
        if len(merged_class_proto.overloaded_methods) > 0:
            assert_merged_overloaded_functions_to_proto(merged_class_proto.overloaded_methods,
                                                        original_class_symbol.overloaded_methods)
        assert merged_class_proto.valid_for == original_class_symbol.valid_for


def assert_merged_function_symbol_to_proto(merged_functions_proto, merged_functions):
    assert len(merged_functions_proto) == len(merged_functions)
    for merged_function_proto in merged_functions_proto:
        matching_function_symbol = merged_functions[merged_function_proto.fully_qualified_name]
        assert len(matching_function_symbol) == 1
        original_function_symbol = matching_function_symbol[0]
        assert merged_function_proto.name == original_function_symbol.function_symbol.name
        assert merged_function_proto.fully_qualified_name == original_function_symbol.function_symbol.fullname
        assert merged_function_proto.has_decorators == original_function_symbol.function_symbol.has_decorators
        assert (merged_function_proto.resolved_decorator_names
                == original_function_symbol.function_symbol.resolved_decorator_names)
        assert merged_function_proto.is_abstract == original_function_symbol.function_symbol.is_abstract
        assert merged_function_proto.is_asynchronous == original_function_symbol.function_symbol.is_asynchronous
        assert merged_function_proto.is_final == original_function_symbol.function_symbol.is_final
        assert merged_function_proto.is_overload == original_function_symbol.function_symbol.is_overload
        assert merged_function_proto.is_property == original_function_symbol.function_symbol.is_property
        assert merged_function_proto.is_static == original_function_symbol.function_symbol.is_static
        assert merged_function_proto.is_class_method == original_function_symbol.function_symbol.is_class_method
        assert merged_function_proto.valid_for == original_function_symbol.valid_for


def assert_merged_overloaded_functions_to_proto(merged_overloaded_functions_proto, merged_overloaded_functions):
    assert len(merged_overloaded_functions_proto) == len(merged_overloaded_functions)
    for merged_overloaded_function_proto in merged_overloaded_functions_proto:
        matching_overloaded_function = merged_overloaded_functions[merged_overloaded_function_proto.fullname]
        assert len(matching_overloaded_function) == 1
        original_overloaded_function = matching_overloaded_function[0]
        assert merged_overloaded_function_proto.name == original_overloaded_function.overloaded_function_symbol.name
        assert (merged_overloaded_function_proto.fullname
                == original_overloaded_function.overloaded_function_symbol.fullname)
        assert (len(merged_overloaded_function_proto.definitions)
                == len(original_overloaded_function.overloaded_function_symbol.definitions))


def assert_abc_merged_module(merged_modules, expected_valid_for):
    assert len(merged_modules) == 1
    abc_merged_symbol = merged_modules["abc"]
    assert isinstance(abc_merged_symbol, MergedModuleSymbol)
    assert abc_merged_symbol.fullname == "abc"
    assert ([c for c in abc_merged_symbol.classes]
            == ['abc.ABCMeta', 'abc.abstractproperty', 'abc.ABC'])
    for merged_class_proto in abc_merged_symbol.classes.values():
        assert len(merged_class_proto) == 1
        assert merged_class_proto[0].valid_for == expected_valid_for
    assert ([f for f in abc_merged_symbol.functions]
            == ['abc.abstractmethod', 'abc.abstractstaticmethod', 'abc.abstractclassmethod', 'abc.get_cache_token'])
    for f in abc_merged_symbol.functions.values():
        assert len(f) == 1
        assert f[0].valid_for == expected_valid_for
    assert len(abc_merged_symbol.overloaded_functions) == 0


def test_alias(fake_module_36_38):
    fake_module_36 = symbols.ModuleSymbol(fake_module_36_38[0])
    fake_module_38 = symbols.ModuleSymbol(fake_module_36_38[1])
    merged_modules = symbols_merger.merge_modules({"fakemodule"}, {"36": {"fakemodule": fake_module_36},
                                                                   "38": {"fakemodule": fake_module_38}})
    merged_fakemodule_module = merged_modules['fakemodule']
    sys_flags = merged_fakemodule_module.vars["sys.flags"][0].var_symbol
    assert sys_flags is not None
    assert sys_flags.name == "my_flags"
