import unittest

import mypy.nodes as mpn

from serializer import typeshed_serializer, symbols


class TypeshedSerializerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build_result = typeshed_serializer.walk_typeshed_stdlib()

    def test_build_mypy_model(self):
        self.assertIsNotNone(self.build_result)

    def test_module_symbol(self):
        abc_module = self.build_result.files.get("abc")
        module_symbol = symbols.ModuleSymbol(abc_module)
        self.assertEqual(module_symbol.fullname, "abc")
        self.assertEqual(len(module_symbol.classes), 3)
        self.assertEqual(len(module_symbol.functions), 4)

    def test_class_symbol(self):
        mypy_cmd_module = self.build_result.files.get("cmd")
        mypy_cmd_class = mypy_cmd_module.names.get("Cmd")
        cmd_class_symbol = symbols.ClassSymbol(mypy_cmd_class.node)
        self.assertEqual(cmd_class_symbol.fullname, "cmd.Cmd")
        self.assertEqual(cmd_class_symbol.name, "Cmd")
        self.assertEqual(cmd_class_symbol.super_classes, ["builtins.object"])
        self.assertEqual(len(cmd_class_symbol.mro), 0)

    def test_function_symbol(self):
        mypy_cmd_module = self.build_result.files.get("cmd")
        mypy_cmd_class_node = mypy_cmd_module.names.get("Cmd").node
        self.assertIsInstance(mypy_cmd_class_node, mpn.TypeInfo)
        mypy_completenames_method_node = mypy_cmd_class_node.names.get("completenames").node
        self.assertIsInstance(mypy_completenames_method_node, mpn.FuncDef)
        completenames_method_symbol = symbols.FunctionSymbol(mypy_completenames_method_node)
        self.assertEqual(completenames_method_symbol.name, "completenames")
        self.assertEqual(completenames_method_symbol.fullname, "cmd.Cmd.completenames")
        self.assertFalse(completenames_method_symbol.has_decorators)
        self.assertFalse(completenames_method_symbol.is_coroutine)
        self.assertFalse(completenames_method_symbol.is_async_generator)
        self.assertEqual(completenames_method_symbol.return_type.pretty_printed_name, "builtins.list[builtins.str]")
        self.assertEqual(len(completenames_method_symbol.resolved_decorator_names), 0)
        self.assertEqual(completenames_method_symbol.fullname, "cmd.Cmd.completenames")

        args = completenames_method_symbol.arguments
        self.assertEqual(len(args), 3)


if __name__ == '__main__':
    unittest.main()
