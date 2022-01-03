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

from unittest import mock

import pytest

from serializer import symbols
import mypy.nodes as mpn


def test_module_symbol(typeshed_stdlib):
    abc_module = typeshed_stdlib.files.get("abc")
    module_symbol = symbols.ModuleSymbol(abc_module)
    assert module_symbol.fullname == "abc"
    assert len(module_symbol.classes) == 5
    assert len(module_symbol.functions) == 4

    pb_module = module_symbol.to_proto()
    assert pb_module.fully_qualified_name == "abc"
    assert len(pb_module.classes) == 5
    assert len(pb_module.functions) == 4


def test_class_symbol(typeshed_stdlib):
    mypy_cmd_module = typeshed_stdlib.files.get("cmd")
    mypy_cmd_class = mypy_cmd_module.names.get("Cmd")
    cmd_class_symbol = symbols.ClassSymbol(mypy_cmd_class.node)
    assert cmd_class_symbol.fullname == "cmd.Cmd"
    assert cmd_class_symbol.name == "Cmd"
    assert cmd_class_symbol.super_classes == ["builtins.object"]

    pb_class_symbol = cmd_class_symbol.to_proto()
    assert pb_class_symbol.name == "Cmd"
    assert pb_class_symbol.fully_qualified_name == "cmd.Cmd"
    assert cmd_class_symbol.super_classes == ["builtins.object"]


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

    pb_func = completenames_method_symbol.to_proto()
    assert pb_func.name == "completenames"
    assert pb_func.fully_qualified_name == "cmd.Cmd.completenames"
    assert not pb_func.has_decorators
    assert not pb_func.is_asynchronous
    assert pb_func.return_annotation.pretty_printed_name == "builtins.list[builtins.str]"
    assert len(pb_func.resolved_decorator_names) == 0


def test_overloaded_functions(typeshed_stdlib):
    sys_module_symbol = symbols.ModuleSymbol(typeshed_stdlib.files.get("sys"))
    overloaded_functions = sys_module_symbol.overloaded_functions
    assert len(overloaded_functions) == 1
    overloaded_func = overloaded_functions[0]
    assert overloaded_func.name == "getsizeof"
    assert overloaded_func.fullname == "sys.getsizeof"
    assert len(overloaded_func.definitions) == 2

    overloaded_func_proto = overloaded_func.to_proto()
    assert overloaded_func_proto.name == "getsizeof"
    assert overloaded_func_proto.fullname == "sys.getsizeof"
    assert len(overloaded_func_proto.definitions) == 2


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
