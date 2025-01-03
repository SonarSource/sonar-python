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

import logging
import os
from enum import Enum
from typing import List, Union

import mypy.types as mpt
import mypy.nodes as mpn

from serializer.proto_out import symbols_pb2

CURRENT_PATH = os.path.dirname(__file__)

logger = logging.getLogger(__name__)

DEFAULT_EXPORTED_VARS = ["__name__", "__doc__", "__file__", "__package__"]
SONAR_CUSTOM_BASE_CLASS = "SonarPythonAnalyzerFakeStub.CustomStubBase"


class ParamKind(Enum):
    POSITIONAL_ONLY = 0
    POSITIONAL_OR_KEYWORD = 1
    KEYWORD_ONLY = 2
    VAR_KEYWORD = 3
    VAR_POSITIONAL = 4


class TypeKind(Enum):
    INSTANCE = 0
    UNION = 1
    TYPE = 2
    TUPLE = 3
    TYPE_VAR = 4
    ANY = 5
    NONE = 6
    TYPE_ALIAS = 7
    CALLABLE = 8
    LITERAL = 9
    UNINHABITED = 10
    UNBOUND = 11
    TYPED_DICT = 12


class TypeDescriptor:
    def __init__(self, _type: mpt.Type, visited=None):
        if visited is None:
            visited = set()
        self.args = []
        self.fully_qualified_name = None
        self.kind = None
        self.is_unknown = False
        self.pretty_printed_name = "Unknown"
        # kind, fqn, pretty printed name, arguments
        if isinstance(_type, mpt.Instance):
            self.kind = TypeKind.INSTANCE
            if _type.args is not None and len(_type.args) > 0:
                items = [TypeDescriptor(t, visited) for t in _type.args]
                item_names = [i.pretty_printed_name for i in items]
                self.args.extend(items)
                self.pretty_printed_name = f"{_type.type.fullname}[{','.join(item_names)}]"
                self.fully_qualified_name = _type.type.fullname
            else:
                self.pretty_printed_name = _type.type.fullname
                self.fully_qualified_name = _type.type.fullname
        elif isinstance(_type, mpt.UnionType):
            self.kind = TypeKind.UNION
            items = [TypeDescriptor(t, visited) for t in _type.items]
            self.args.extend(items)
            item_names = [i.pretty_printed_name for i in items]
            self.pretty_printed_name = f"Union[{','.join(item_names)}]"
        elif isinstance(_type, mpt.TypeType):
            self.kind = TypeKind.TYPE
            item = TypeDescriptor(_type.item, visited)
            self.args.append(item)
            self.pretty_printed_name = f"Type[{item.pretty_printed_name}]"
            self.fully_qualified_name = "type"
        elif isinstance(_type, mpt.TupleType):
            self.kind = TypeKind.TUPLE
            items = [TypeDescriptor(t, visited) for t in _type.items]
            if any(item.is_unknown for item in items):
                self.kind = None
                self.is_unknown = True
            else:
                item_names = [i.pretty_printed_name for i in items]
                self.args.extend(items)
                self.pretty_printed_name = f"Tuple[{','.join(item_names)}]"
        elif isinstance(_type, mpt.TypeVarType):
            self.kind = TypeKind.TYPE_VAR
            self.pretty_printed_name = _type.fullname
            upper_bound = TypeDescriptor(_type.upper_bound)
            self.args.append(upper_bound)
            self.fully_qualified_name = upper_bound.fully_qualified_name
        elif isinstance(_type, mpt.AnyType):
            self.kind = TypeKind.ANY
            self.pretty_printed_name = "Any"
        elif isinstance(_type, mpt.NoneType):
            self.kind = TypeKind.NONE
            self.pretty_printed_name = "None"
        elif isinstance(_type, mpt.TypeAliasType):
            self.kind = TypeKind.TYPE_ALIAS
            alias = _type.alias
            if alias.target not in visited:
                visited.add(alias.target)
                target = TypeDescriptor(alias.target, visited)
                self.args.append(target)
                self.pretty_printed_name = f"TypeAlias[{target.pretty_printed_name}]"
                self.fully_qualified_name = _type.alias.fullname
            else:
                self.kind = None
                self.is_unknown = True
        elif isinstance(_type, mpt.CallableType):
            self.kind = TypeKind.CALLABLE
            fallback = TypeDescriptor(_type.fallback, visited)
            self.args.append(fallback)
            self.pretty_printed_name = f"CallableType[{fallback.pretty_printed_name}]"
        elif isinstance(_type, mpt.LiteralType):
            self.kind = TypeKind.LITERAL
            fallback = TypeDescriptor(_type.fallback, visited)
            self.args.append(fallback)
            self.pretty_printed_name = f"Literal[{fallback.pretty_printed_name}]"
        elif isinstance(_type, mpt.UninhabitedType):
            self.kind = TypeKind.UNINHABITED
            self.pretty_printed_name = "NoReturn"
        elif isinstance(_type, mpt.UnboundType):
            self.kind = TypeKind.UNBOUND
            self.pretty_printed_name = f"UnboundType[{_type.name}]"
        elif isinstance(_type, mpt.TypedDictType):
            self.kind = TypeKind.TYPED_DICT
            # TODO: check in items for key/type mapping
            self.pretty_printed_name = "TypedDict"
        elif isinstance(_type, mpt.Overloaded):
            self.kind = TypeKind.CALLABLE
            fallback = TypeDescriptor(_type.fallback, visited)
            self.fully_qualified_name = fallback.fully_qualified_name
            self.args.append(fallback)
            self.pretty_printed_name = f"CallableType[{fallback.pretty_printed_name}]"
        else:
            # this can happen when there is a var symbol assigned to an overload symbol
            self.is_unknown = True

    def to_proto(self) -> symbols_pb2.Type:
        pb_type = symbols_pb2.Type()
        if self.is_unknown:
            return pb_type
        pb_type.pretty_printed_name = self.pretty_printed_name
        if self.kind is not None:
            pb_type.kind = symbols_pb2.TypeKind.Value(self.kind.name)
        if self.fully_qualified_name is not None:
            pb_type.fully_qualified_name = self.fully_qualified_name
        for arg in self.args:
            pb_type.args.append(arg.to_proto())
        return pb_type


class ParameterSymbol:
    def __init__(self, kind, name, _type, param):
        self.name = param
        self.kind = None
        self.type_annotation = None
        self.has_default = kind in [mpn.ARG_OPT, mpn.ARG_NAMED_OPT]
        if name is None:
            self.kind = ParamKind.POSITIONAL_ONLY
        if kind == mpn.ARG_STAR:
            self.kind = ParamKind.VAR_POSITIONAL
        if kind == mpn.ARG_STAR2:
            self.kind = ParamKind.VAR_KEYWORD
        if kind in [mpn.ARG_NAMED, mpn.ARG_NAMED_OPT]:
            self.kind = ParamKind.KEYWORD_ONLY
        if self.kind is None:
            self.kind = ParamKind.POSITIONAL_OR_KEYWORD
        if _type is not None:
            self.type_annotation = TypeDescriptor(_type)

    def to_proto(self) -> symbols_pb2.ParameterSymbol:
        pb_parameter = symbols_pb2.ParameterSymbol()
        if self.name is not None:
            pb_parameter.name = self.name
        pb_parameter.kind = symbols_pb2.ParameterKind.Value(self.kind.name)
        pb_parameter.has_default = self.has_default
        if self.type_annotation is not None:
            pb_parameter.type_annotation.CopyFrom(self.type_annotation.to_proto())
        return pb_parameter


class OverloadedFunctionSymbol:
    def __init__(self, overloaded_func_def: mpn.OverloadedFuncDef, name: str = None):
        self.name = overloaded_func_def.name if name is None else name
        self.fullname = overloaded_func_def.fullname
        self.definitions = []
        for item in overloaded_func_def.items:
            self.add_overloaded_func_definition(item)
        if len(self.definitions) < 2:
            # Consider unanalyzed items if analyzed definitions are missing
            if len(overloaded_func_def.unanalyzed_items) > 0:
                logger.warning(f'Overloaded function definitions of '
                               f'"{overloaded_func_def.fullname}" are missing: falling back on unanalyzed items.')
            for item in overloaded_func_def.unanalyzed_items:
                self.add_overloaded_func_definition(item)
        if len(self.definitions) < 2:
            raise RuntimeError("Overloaded function symbol should contain at least 2 definitions.")

    def add_overloaded_func_definition(self, item):
        if isinstance(item, mpn.FuncDef):
            # Should not happen?
            self.definitions.append(FunctionSymbol(item))
        if isinstance(item, mpn.Decorator):
            self.definitions.append(FunctionSymbol(item.func, decorators=item.original_decorators))

    def __eq__(self, other):
        return isinstance(other, OverloadedFunctionSymbol) and self.to_proto() == other.to_proto()

    def to_proto(self) -> symbols_pb2.OverloadedFunctionSymbol:
        pb_overloaded_func = symbols_pb2.OverloadedFunctionSymbol()
        pb_overloaded_func.name = self.name
        pb_overloaded_func.fullname = self.fullname
        for definition in self.definitions:
            pb_overloaded_func.definitions.append(definition.to_proto())
        return pb_overloaded_func


class FunctionSymbol:
    def __init__(self, func_def: mpn.FuncDef, decorators=None, name: str = None):
        self.name = func_def.name if name is None else name
        self.fullname = func_def.fullname
        self.return_type = extract_return_type(func_def)
        self.parameters = extract_parameters(func_def)
        self.has_decorators = func_def.is_decorated
        self.is_abstract = False if func_def.abstract_status == mpn.NOT_ABSTRACT else True
        self.is_asynchronous = func_def.is_async_generator or func_def.is_awaitable_coroutine or func_def.is_coroutine
        self.is_final = func_def.is_final
        self.is_overload = func_def.is_overload
        self.is_property = func_def.is_property
        self.is_static = func_def.is_static
        self.is_class_method = func_def.is_class
        self.resolved_decorator_names = []
        if self.has_decorators and decorators is not None:
            for dec in decorators:
                decorator_name = get_decorator_name(dec)
                if decorator_name is not None:
                    self.resolved_decorator_names.append(decorator_name)

    def __eq__(self, other):
        return isinstance(other, FunctionSymbol) and self.to_proto() == other.to_proto()

    def to_proto(self) -> symbols_pb2.FunctionSymbol:
        pb_func = symbols_pb2.FunctionSymbol()
        pb_func.name = self.name
        pb_func.fully_qualified_name = self.fullname
        pb_func.has_decorators = self.has_decorators
        pb_func.resolved_decorator_names.extend(self.resolved_decorator_names)
        pb_func.is_abstract = self.is_abstract
        pb_func.is_asynchronous = self.is_asynchronous
        pb_func.is_final = self.is_final
        pb_func.is_overload = self.is_overload
        pb_func.is_property = self.is_property
        pb_func.is_static = self.is_static
        pb_func.is_class_method = self.is_class_method
        if self.return_type is not None:
            pb_func.return_annotation.CopyFrom(self.return_type.to_proto())
        for parameter in self.parameters:
            pb_func.parameters.append(parameter.to_proto())
        return pb_func


class ClassSymbol:
    def __init__(self, type_info: mpn.TypeInfo, name: str = None):
        self.name = type_info.name if name is None else name
        self.fullname = type_info.fullname
        self.super_classes = []
        self.methods = []
        self.overloaded_methods = []
        self.vars = []
        self.is_enum = type_info.is_enum
        self.is_generic = type_info.is_generic()
        self.is_protocol = type_info.is_protocol
        self.metaclass_name = None
        for base in type_info.bases:
            if isinstance(base, mpt.Instance):
                self.super_classes.append(base.type.fullname)
        for key in type_info.names:
            name = type_info.names.get(key)
            node = name.node
            if isinstance(node, mpn.FuncDef):
                self.methods.append(FunctionSymbol(node))
            elif isinstance(node, mpn.Decorator):
                self.methods.append(FunctionSymbol(node.func, decorators=node.original_decorators))
            elif isinstance(node, mpn.OverloadedFuncDef):
                self.overloaded_methods.append(OverloadedFunctionSymbol(node))
            elif isinstance(node, mpn.Var) and node.name not in DEFAULT_EXPORTED_VARS:
                self.vars.append(VarSymbol.from_var(node))
        class_def = type_info.defn
        self.has_metaclass = class_def.metaclass is not None
        if class_def.metaclass is not None:
            self.has_metaclass = True
            if isinstance(class_def.metaclass, mpn.NameExpr) or isinstance(class_def.metaclass, mpn.MemberExpr):
                self.metaclass_name = class_def.metaclass.fullname
        self.has_decorators = len(class_def.decorators) > 0

    def __eq__(self, other):
        if not isinstance(other, ClassSymbol):
            return False
        return (self.name == other.name
                and self.fullname == other.fullname
                and self.super_classes == other.super_classes
                and self.is_enum == other.is_enum
                and self.is_generic == other.is_generic
                and self.is_protocol == other.is_protocol
                and self.metaclass_name == other.metaclass_name
                and self.has_decorators == other.has_decorators)

    def to_proto(self) -> symbols_pb2.ClassSymbol:
        pb_class = symbols_pb2.ClassSymbol()
        pb_class.name = self.name
        pb_class.fully_qualified_name = self.fullname
        pb_class.super_classes.extend(self.super_classes)
        pb_class.has_decorators = self.has_decorators
        pb_class.has_metaclass = self.has_metaclass
        pb_class.is_enum = self.is_enum
        pb_class.is_generic = self.is_generic
        pb_class.is_protocol = self.is_protocol
        if self.metaclass_name is not None:
            pb_class.metaclass_name = self.metaclass_name
        for method in self.methods:
            pb_class.methods.append(method.to_proto())
        for overloaded_method in self.overloaded_methods:
            pb_class.overloaded_methods.append(overloaded_method.to_proto())
        for var in self.vars:
            pb_class.attributes.append(var.to_proto())
        return pb_class


class VarSymbol:
    def __init__(self, name: str, fullname: str, is_imported_module=False, type_descriptor: TypeDescriptor = None):
        self.name = name
        self.fullname = fullname
        self.is_imported_module = is_imported_module
        self.type = type_descriptor

    @classmethod
    def from_var(cls, var: mpn.Var, name: str = None):
        return cls(var.name if name is None else name, var.fullname,
                   type_descriptor=TypeDescriptor(var.type) if var.type else None)

    def __eq__(self, other):
        return isinstance(other, VarSymbol) and self.to_proto() == other.to_proto()

    def to_proto(self) -> symbols_pb2.VarSymbol:
        pb_var = symbols_pb2.VarSymbol()
        pb_var.name = self.name
        pb_var.fully_qualified_name = self.fullname
        if self.type is not None:
            pb_var.type_annotation.CopyFrom(self.type.to_proto())
        pb_var.is_imported_module = self.is_imported_module
        return pb_var


class ModuleSymbol:
    def __init__(self, mypy_file: mpn.MypyFile):
        self.fullname = mypy_file.fullname
        self.classes = []
        self.functions = []
        self.overloaded_functions = []
        self.vars = []
        private_imports = set()
        for elem in mypy_file.imports:
            # imports without aliases are considered private in Typeshed convention
            if isinstance(elem, mpn.Import):
                for _id, alias in elem.ids:
                    if _id != alias:
                        private_imports.add(_id)
            if isinstance(elem, mpn.ImportFrom):
                for _id, alias in elem.names:
                    if _id != alias:
                        private_imports.add(_id)
        for key in mypy_file.names:
            name = mypy_file.names.get(key)
            if key in private_imports and not name.fullname.startswith(mypy_file.fullname):
                continue
            if name.fullname == SONAR_CUSTOM_BASE_CLASS:
                # Ignore custom stub name
                continue
            symbol_table_node = name.node
            if isinstance(symbol_table_node, mpn.FuncDef):
                self.functions.append(FunctionSymbol(symbol_table_node, name=key))
            if isinstance(symbol_table_node, mpn.OverloadedFuncDef):
                self.overloaded_functions.append(OverloadedFunctionSymbol(symbol_table_node, name=key))
            if isinstance(symbol_table_node, mpn.TypeInfo):
                self.classes.append(ClassSymbol(symbol_table_node, name=key))
            if isinstance(symbol_table_node, mpn.Var) and symbol_table_node.name not in DEFAULT_EXPORTED_VARS:
                self.vars.append(VarSymbol.from_var(symbol_table_node, name=key))
            if isinstance(symbol_table_node, mpn.MypyFile):
                module_name = symbol_table_node.fullname
                if module_name != "builtins":
                    self.vars.append(VarSymbol(key, module_name, is_imported_module=True))

    def to_proto(self) -> symbols_pb2.ModuleSymbol:
        pb_module = symbols_pb2.ModuleSymbol()
        pb_module.fully_qualified_name = self.fullname
        for cls in self.classes:
            pb_module.classes.append(cls.to_proto())
        for func in self.functions:
            pb_module.functions.append(func.to_proto())
        for overloaded_func in self.overloaded_functions:
            pb_module.overloaded_functions.append(overloaded_func.to_proto())
        for var in self.vars:
            pb_module.vars.append(var.to_proto())
        return pb_module


class MergedFunctionSymbol:
    def __init__(self, function_symbol: FunctionSymbol, valid_for: List[str]):
        self.function_symbol = function_symbol
        self.valid_for = valid_for

    def to_proto(self) -> symbols_pb2.FunctionSymbol:
        pb_func = self.function_symbol.to_proto()
        for elem in self.valid_for:
            pb_func.valid_for.append(elem)
        return pb_func


class MergedOverloadedFunctionSymbol:
    def __init__(self, overloaded_function_symbol: OverloadedFunctionSymbol, valid_for: List[str]):
        self.overloaded_function_symbol = overloaded_function_symbol
        self.valid_for = valid_for

    def to_proto(self) -> symbols_pb2.FunctionSymbol:
        pb_func = self.overloaded_function_symbol.to_proto()
        for elem in self.valid_for:
            pb_func.valid_for.append(elem)
        return pb_func


class MergedClassSymbol:
    def __init__(self, reference_class_symbols: ClassSymbol, merged_methods, merged_overloaded_methods,
                 merged_attributes, valid_for: List[str]):
        # nested class symbols functions are not relevant anymore
        self.class_symbol = reference_class_symbols
        self.methods = merged_methods
        self.overloaded_methods = merged_overloaded_methods
        self.vars = merged_attributes
        self.valid_for = valid_for

    def to_proto(self) -> symbols_pb2.ClassSymbol:
        pb_class = symbols_pb2.ClassSymbol()
        pb_class.name = self.class_symbol.name
        pb_class.fully_qualified_name = self.class_symbol.fullname
        pb_class.super_classes.extend(self.class_symbol.super_classes)
        pb_class.has_decorators = self.class_symbol.has_decorators
        pb_class.has_metaclass = self.class_symbol.has_metaclass
        pb_class.is_enum = self.class_symbol.is_enum
        pb_class.is_generic = self.class_symbol.is_generic
        pb_class.is_protocol = self.class_symbol.is_protocol
        if self.class_symbol.metaclass_name is not None:
            pb_class.metaclass_name = self.class_symbol.metaclass_name
        for method in self.methods:
            for elem in self.methods[method]:
                pb_class.methods.append(elem.to_proto())
        for overloaded_func in self.overloaded_methods:
            for elem in self.overloaded_methods[overloaded_func]:
                pb_class.overloaded_methods.append(elem.to_proto())
        for var in self.vars:
            for elem in self.vars[var]:
                pb_class.attributes.append(elem.to_proto())
        for elem in self.valid_for:
            pb_class.valid_for.append(elem)
        return pb_class


class MergedVarSymbol:
    def __init__(self, var_symbol: VarSymbol, valid_for: List[str]):
        self.var_symbol = var_symbol
        self.valid_for = valid_for

    def to_proto(self) -> symbols_pb2.VarSymbol:
        pb_var = self.var_symbol.to_proto()
        for elem in self.valid_for:
            pb_var.valid_for.append(elem)
        return pb_var


class MergedModuleSymbol:
    def __init__(self, fullname, classes, functions, overloaded_functions, variables):
        self.fullname = fullname
        self.classes = classes
        self.functions = functions
        self.overloaded_functions = overloaded_functions
        self.vars = variables

    def to_proto(self):
        pb_module = symbols_pb2.ModuleSymbol()
        pb_module.fully_qualified_name = self.fullname
        for cls in self.classes:
            for elem in self.classes[cls]:
                pb_module.classes.append(elem.to_proto())
        for func in self.functions:
            for elem in self.functions[func]:
                pb_module.functions.append(elem.to_proto())
        for overloaded_func in self.overloaded_functions:
            for elem in self.overloaded_functions[overloaded_func]:
                pb_module.overloaded_functions.append(elem.to_proto())
        for var in self.vars:
            for elem in self.vars[var]:
                pb_module.vars.append(elem.to_proto())
        return pb_module


def get_decorator_name(dec: mpn.Node):
    if isinstance(dec, mpn.NameExpr):
        # decorator full name might not be the actual fully qualified name if it could not be resolved
        # TODO: handle "None" case and check for fallbacks
        if dec.name is not None:
            return dec.name
    if isinstance(dec, mpn.MemberExpr):
        prefix = get_decorator_name(dec.expr)
        if prefix is not None and dec.name is not None:
            return f"{prefix}.{dec.name}"
    return None


def extract_parameters(func_def: mpn.FuncDef):
    arguments = []
    func_type = func_def.type
    if not isinstance(func_type, mpt.CallableType):
        # Missing type info: only basic information available
        for kind, name in zip(func_def.arg_kinds, func_def.arg_names):
            arguments.append(ParameterSymbol(kind, name, None, name))
        return arguments
    arg_kinds = func_type.arg_kinds
    arg_names = func_type.arg_names
    arg_types = func_type.arg_types
    # param names are actual names of the parameters
    # arg names can be None for positional only arguments
    param_names = func_def.arg_names
    for kind, name, _type, param in zip(arg_kinds, arg_names, arg_types, param_names):
        # Assumption to validate: all lists have always the same length == to the number of params
        arguments.append(ParameterSymbol(kind, name, _type, param))
    return arguments


def extract_return_type(func_def: mpn.FuncDef):
    func_type = func_def.type
    if not isinstance(func_type, mpt.CallableType):
        # Missing type info: no return type is available
        return None
    return TypeDescriptor(func_type.ret_type)


def save_module(ms: Union[ModuleSymbol, MergedModuleSymbol], dir_name="stdlib_protobuf",
                is_debug=False, debug_dir="output"):
    ms_pb = ms.to_proto()
    save_dir = dir_name if not is_debug else f"../{debug_dir}"
    save_string = ms_pb.SerializeToString() if not is_debug else str(ms_pb)
    open_mode = "wb" if not is_debug else "w"
    save_dir_path = os.path.join(CURRENT_PATH, save_dir)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    save_name = ms.fullname if not is_python_2_only_exception(ms) else f"2@{ms.fullname}"
    with open(f"{save_dir_path}/{save_name}.protobuf", open_mode) as f:
        f.write(save_string)


def is_python_2_only_exception(ms) -> bool:
    """ This methods aims to flag some Python 2 modules whose name differ from their Python 3 counterpart
    by capitalization only. This is done to avoid conflicts in the saved file for OS which are not case sensitive
    (e.g Windows and macOS)
    """
    if (not isinstance(ms, MergedModuleSymbol)
            or ms.fullname not in ['ConfigParser', 'Queue', 'SocketServer']):
        return False
    return True
