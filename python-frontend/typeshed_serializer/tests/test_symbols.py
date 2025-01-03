#
# SonarQube Python Plugin
# Copyright (C) 2011-2025 SonarSource SA
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

from unittest import mock

import pytest

from serializer import symbols
import mypy.nodes as mpn

OBJECT_FQN = "builtins.object"


def test_module_symbol(typeshed_stdlib):
    abc_module = typeshed_stdlib.files.get("abc")
    module_symbol = symbols.ModuleSymbol(abc_module)
    assert module_symbol.fullname == "abc"
    assert len(module_symbol.classes) == 3
    assert len(module_symbol.functions) == 4

    pb_module = module_symbol.to_proto()
    assert pb_module.fully_qualified_name == "abc"
    assert len(pb_module.classes) == 3
    assert len(pb_module.functions) == 4
    imported_modules = [imported_module for imported_module in pb_module.vars if
                        imported_module.is_imported_module is True]
    assert len(imported_modules) == 0

    os_module = typeshed_stdlib.files.get("os")
    pb_module = symbols.ModuleSymbol(os_module).to_proto()
    imported_modules = [imported_module for imported_module in pb_module.vars if
                        imported_module.is_imported_module is True]
    assert len(imported_modules) == 2
    imported_modules = map(lambda m: (m.fully_qualified_name, m.name), imported_modules)
    assert ("os.path", "_path") in imported_modules
    assert ("os.path", "path") in imported_modules


def test_self_defined_symbol(fake_module_typing_features):
    module_symbol = symbols.ModuleSymbol(fake_module_typing_features)
    assert module_symbol.fullname == "fakemodule_typing_features"
    assert len(module_symbol.classes) == 2
    assert len(module_symbol.functions) == 1

    pb_module = module_symbol.to_proto()
    assert pb_module.fully_qualified_name == "fakemodule_typing_features"
    assert len(pb_module.classes) == 2
    assert len(pb_module.functions) == 1


def test_typevar(fake_module_typing_features):
    module_symbol = symbols.ModuleSymbol(fake_module_typing_features)
    for func_def in module_symbol.overloaded_functions[0].definitions:
        assert func_def.return_type.fully_qualified_name == "fakemodule_typing_features.MyClassWithTypeVar"
    func = module_symbol.functions[0]
    assert func.return_type.fully_qualified_name == OBJECT_FQN


def test_class_symbol(typeshed_stdlib):
    mypy_cmd_module = typeshed_stdlib.files.get("cmd")
    mypy_cmd_class = mypy_cmd_module.names.get("Cmd")
    cmd_class_symbol = symbols.ClassSymbol(mypy_cmd_class.node)
    assert cmd_class_symbol.fullname == "cmd.Cmd"
    assert cmd_class_symbol.name == "Cmd"
    assert cmd_class_symbol.super_classes == [OBJECT_FQN]
    assert len(cmd_class_symbol.methods) == 18
    assert len(cmd_class_symbol.vars) == 17

    pb_class_symbol = cmd_class_symbol.to_proto()
    assert pb_class_symbol.name == "Cmd"
    assert pb_class_symbol.fully_qualified_name == "cmd.Cmd"
    assert cmd_class_symbol.super_classes == [OBJECT_FQN]


def test_type_return(typeshed_stdlib):
    mypy_collections_module = typeshed_stdlib.files.get("collections")
    mypy_namedtuple_function = mypy_collections_module.names.get("namedtuple")

    namedtuple_function_symbol = symbols.FunctionSymbol(mypy_namedtuple_function.node)
    return_type = namedtuple_function_symbol.return_type
    assert return_type.fully_qualified_name == "type"
    assert return_type.pretty_printed_name == "Type[builtins.tuple[Any]]"
    assert return_type.args[0].fully_qualified_name == "builtins.tuple"
    assert return_type.args[0].pretty_printed_name == "builtins.tuple[Any]"


def test_class_symbol_metaclass(typeshed_stdlib):
    io_module = typeshed_stdlib.files.get("io")
    iobase_class = io_module.names.get("IOBase")
    iobase_class_symbol = symbols.ClassSymbol(iobase_class.node)
    assert iobase_class_symbol.has_metaclass
    assert iobase_class_symbol.metaclass_name == "abc.ABCMeta"

    name_meta_class = io_module.names.get("NameMeta")
    name_meta_class_symbol = symbols.ClassSymbol(name_meta_class.node)
    assert name_meta_class_symbol.has_metaclass
    assert name_meta_class_symbol.metaclass_name == "abc.ABCMeta"

    str_meta_class = io_module.names.get("StrMeta")
    str_meta_class_symbol = symbols.ClassSymbol(str_meta_class.node)
    assert str_meta_class_symbol.has_metaclass
    assert str_meta_class_symbol.metaclass_name is None


def test_function_symbol(typeshed_stdlib):
    mypy_cmd_module = typeshed_stdlib.files.get("cmd")

    mypy_cmd_class_node = mypy_cmd_module.names.get("Cmd").node
    assert isinstance(mypy_cmd_class_node, mpn.TypeInfo)

    mypy_completenames_method_node = mypy_cmd_class_node.names.get("completenames").node
    assert isinstance(mypy_completenames_method_node, mpn.FuncDef)

    completenames_method_symbol = symbols.FunctionSymbol(mypy_completenames_method_node)
    assert completenames_method_symbol.name == "completenames"
    assert completenames_method_symbol.fullname == "cmd.Cmd.completenames"
    assert not completenames_method_symbol.has_decorators
    assert not completenames_method_symbol.is_asynchronous
    assert completenames_method_symbol.return_type.pretty_printed_name == "builtins.list[builtins.str]"
    assert len(completenames_method_symbol.resolved_decorator_names) == 0
    assert completenames_method_symbol.fullname == "cmd.Cmd.completenames"

    args = completenames_method_symbol.parameters
    assert len(args) == 3

    ignored_param = args[2]
    assert not ignored_param.has_default
    assert ignored_param.kind == symbols.ParamKind.VAR_POSITIONAL
    pb_func = completenames_method_symbol.to_proto()

    assert pb_func.name == "completenames"
    assert pb_func.fully_qualified_name == "cmd.Cmd.completenames"
    assert not pb_func.has_decorators
    assert not pb_func.is_asynchronous
    assert pb_func.return_annotation.pretty_printed_name == "builtins.list[builtins.str]"
    assert len(pb_func.resolved_decorator_names) == 0

    mypy_cmd_loop_method_node = mypy_cmd_class_node.names.get("cmdloop").node
    cmd_loop = symbols.FunctionSymbol(mypy_cmd_loop_method_node)
    assert cmd_loop.name == "cmdloop"
    assert len(cmd_loop.parameters) == 2
    self_param = cmd_loop.parameters[0]
    assert not self_param.has_default
    assert self_param.kind == symbols.ParamKind.POSITIONAL_OR_KEYWORD
    intro_param = cmd_loop.parameters[1]
    assert intro_param.has_default
    assert intro_param.kind == symbols.ParamKind.POSITIONAL_OR_KEYWORD


def test_overloaded_functions(typeshed_stdlib):
    sys_module_symbol = symbols.ModuleSymbol(typeshed_stdlib.files.get("subprocess"))
    overloaded_functions = sys_module_symbol.overloaded_functions
    assert len(overloaded_functions) == 2
    overloaded_functions = sorted(overloaded_functions, key=lambda x: x.name)
    overloaded_func = overloaded_functions[0]
    assert overloaded_func.name == "check_output"
    assert overloaded_func.fullname == "subprocess.check_output"
    assert len(overloaded_func.definitions) == 6

    overloaded_func_proto = overloaded_func.to_proto()
    assert overloaded_func_proto.name == "check_output"
    assert overloaded_func_proto.fullname == "subprocess.check_output"
    assert len(overloaded_func_proto.definitions) == 6

    overloaded_func2 = overloaded_functions[1]
    assert overloaded_func2.name == "run"
    assert overloaded_func2.fullname == "subprocess.run"
    assert len(overloaded_func2.definitions) == 6

    overloaded_func_proto2 = overloaded_func2.to_proto()
    assert overloaded_func_proto2.name == "run"
    assert overloaded_func_proto2.fullname == "subprocess.run"
    assert len(overloaded_func_proto2.definitions) == 6


def test_save_module(typeshed_stdlib):
    mock_open = mock.mock_open(read_data='some data from opened file')
    with mock.patch('builtins.open', mock_open):
        abc_module = typeshed_stdlib.files.get("abc")
        module_symbol = symbols.ModuleSymbol(abc_module)
        symbols.save_module(module_symbol)
    mock_open.assert_called_once_with(mock.ANY, 'wb')
    mock_open.return_value.assert_has_calls([mock.call.write(module_symbol.to_proto().SerializeToString())])
    assert mock.call.write(str(module_symbol.to_proto())) not in mock_open.return_value.method_calls


def test_python2_exception():
    queue_symbol = symbols.MergedModuleSymbol("Queue", {}, {}, {}, {})
    other_symbol = symbols.MergedModuleSymbol("other", {}, {}, {}, {})
    assert symbols.is_python_2_only_exception(queue_symbol) is True
    assert symbols.is_python_2_only_exception(other_symbol) is False


def mock_add_definition(self, arg):
    assert isinstance(arg, mpn.Decorator)
    self.definitions.append(arg)


def test_fallback_unanalyzed_items_when_items_are_missing():
    overloaded_func_mock = mock.Mock(mpn.OverloadedFuncDef)
    overloaded_func_mock.items = []
    overloaded_func_mock.fullname = "overloaded_func_mock_fullname"
    func_def_mock1 = mock.Mock(mpn.Decorator)
    func_def_mock2 = mock.Mock(mpn.Decorator)
    overloaded_func_mock.unanalyzed_items = [func_def_mock1, func_def_mock2]

    with mock.patch('serializer.symbols.OverloadedFunctionSymbol.add_overloaded_func_definition', mock_add_definition), \
            mock.patch('logging.Logger.warning') as log_mock:
        overloaded_func_symbol = symbols.OverloadedFunctionSymbol(overloaded_func_mock)
        assert len(overloaded_func_symbol.definitions) == 2
        log_mock.assert_called_with('Overloaded function definitions of "overloaded_func_mock_fullname" are missing: '
                                    'falling back on unanalyzed items.')


def test_error_when_overloaded_definitions_are_missing():
    overloaded_func_mock = mock.Mock(mpn.OverloadedFuncDef)
    overloaded_func_mock.items = []
    overloaded_func_mock.unanalyzed_items = []

    with mock.patch('serializer.symbols.OverloadedFunctionSymbol.add_overloaded_func_definition', mock_add_definition):
        with pytest.raises(RuntimeError) as raised:
            symbols.OverloadedFunctionSymbol(overloaded_func_mock)
        assert raised.value.args[0] == 'Overloaded function symbol should contain at least 2 definitions.'
