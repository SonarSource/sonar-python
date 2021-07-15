import pytest
import mypy.nodes as mpn

from serializer import typeshed_serializer, symbols


@pytest.fixture(scope="module")
def typeshed_stdlib():
    return typeshed_serializer.walk_typeshed_stdlib()


def test_build_mypy_model(typeshed_stdlib):
    assert typeshed_stdlib is not None


def test_module_symbol(typeshed_stdlib):
    abc_module = typeshed_stdlib.files.get("abc")
    module_symbol = symbols.ModuleSymbol(abc_module)
    assert module_symbol.fullname == "abc"
    assert len(module_symbol.classes) == 3
    assert len(module_symbol.functions) == 4


def test_class_symbol(typeshed_stdlib):
    mypy_cmd_module = typeshed_stdlib.files.get("cmd")
    mypy_cmd_class = mypy_cmd_module.names.get("Cmd")
    cmd_class_symbol = symbols.ClassSymbol(mypy_cmd_class.node)
    assert cmd_class_symbol.fullname == "cmd.Cmd"
    assert cmd_class_symbol.name == "Cmd"
    assert cmd_class_symbol.super_classes == ["builtins.object"]
    assert len(cmd_class_symbol.mro) == 0


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
