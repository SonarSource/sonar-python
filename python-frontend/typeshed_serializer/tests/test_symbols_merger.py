from unittest.mock import Mock

from serializer import symbols, symbols_merger
from serializer.symbols import MergedModuleSymbol


def test_build_multiple_python_version(typeshed_stdlib):
    symbols_merger.ts.walk_typeshed_stdlib = Mock(return_value=typeshed_stdlib)
    model_by_version = symbols_merger.build_multiple_python_version()
    assert set(model_by_version.keys()) == {'27', '35', '36', '37', '38', '39'}


def test_merge_multiple_python_versions(typeshed_stdlib):
    symbols_merger.ts.walk_typeshed_stdlib = Mock(return_value=typeshed_stdlib)
    merged_modules = symbols_merger.merge_multiple_python_versions()
    for mod in merged_modules.values():
        assert isinstance(mod, MergedModuleSymbol)
    assert len(merged_modules) == 477


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

    assert len(classes_dict) == 4

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
    assert len(functions_dict) == 4
    common_function_symbols = functions_dict['fakemodule.common_function']
    assert len(common_function_symbols) == 1
    assert common_function_symbols[0].valid_for == ["36", "38"]

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
