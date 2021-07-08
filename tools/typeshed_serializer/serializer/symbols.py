import os
from enum import Enum

import mypy.types as mpt
import mypy.nodes as mpn

from serializer.proto_out import symbols_pb2


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
    def __init__(self, _type: mpt.Type):
        self.args = []
        self.simple_name = None
        # kind, fqn, pretty printed name, arguments
        if isinstance(_type, mpt.Instance):
            self.kind = TypeKind.INSTANCE
            if _type.args is not None and len(_type.args) > 0:
                items = [TypeDescriptor(t) for t in _type.args]
                item_names = [i.pretty_printed_name for i in items]
                self.args.extend(items)
                self.pretty_printed_name = f"{_type.type.fullname}[{','.join(item_names)}]"
                self.simple_name = _type.type.fullname
            else:
                self.pretty_printed_name = _type.type.fullname
                self.simple_name = _type.type.fullname
        elif isinstance(_type, mpt.UnionType):
            self.kind = TypeKind.UNION
            items = [TypeDescriptor(t) for t in _type.items]
            self.args.extend(items)
            item_names = [i.pretty_printed_name for i in items]
            self.pretty_printed_name = f"Union[{','.join(item_names)}]"
        elif isinstance(_type, mpt.TypeType):
            self.kind = TypeKind.TYPE
            item = TypeDescriptor(_type.item)
            self.args.append(item)
            self.pretty_printed_name = f"Type[{item.pretty_printed_name}]"
        elif isinstance(_type, mpt.TupleType):
            self.kind = TypeKind.TUPLE
            items = [TypeDescriptor(t) for t in _type.items]
            item_names = [i.pretty_printed_name for i in items]
            self.args.extend(items)
            self.pretty_printed_name = f"Tuple[{','.join(item_names)}]"
        elif isinstance(_type, mpt.TypeVarType):
            self.kind = TypeKind.TYPE_VAR
            self.pretty_printed_name = _type.fullname
        elif isinstance(_type, mpt.AnyType):
            self.kind = TypeKind.ANY
            self.pretty_printed_name = "Any"
        elif isinstance(_type, mpt.NoneType):
            self.kind = TypeKind.NONE
            self.pretty_printed_name = "None"
        elif isinstance(_type, mpt.TypeAliasType):
            self.kind = TypeKind.TYPE_ALIAS
            alias = _type.alias
            target = TypeDescriptor(alias.target)
            self.args.append(target)
            self.pretty_printed_name = f"TypeAlias[{target.pretty_printed_name}]"
            self.simple_name = _type.alias.fullname
        elif isinstance(_type, mpt.CallableType):
            self.kind = TypeKind.CALLABLE
            fallback = TypeDescriptor(_type.fallback)
            self.args.append(fallback)
            self.pretty_printed_name = f"CallableType[{fallback.pretty_printed_name}]"
        elif isinstance(_type, mpt.LiteralType):
            self.kind = TypeKind.LITERAL
            fallback = TypeDescriptor(_type.fallback)
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
        else:
            # Can this happen?
            self.pretty_printed_name = "#Unknown"

    def to_proto(self) -> symbols_pb2.Type:
        pb_type = symbols_pb2.Type()
        pb_type.pretty_printed_name = self.pretty_printed_name
        pb_type.kind = symbols_pb2.TypeKind.Value(self.kind.name)
        if self.simple_name is not None:
            pb_type.simple_name = self.simple_name
        for arg in self.args:
            pb_type.args.append(arg.to_proto())
        return pb_type


class ParameterSymbol:
    def __init__(self, kind, name, _type, param):
        self.name = param
        self.kind = None
        self.type_annotation = None
        self.has_default = kind in [1, 5]
        if name is None:
            self.kind = ParamKind.POSITIONAL_ONLY
        if kind == 2:
            self.kind = ParamKind.VAR_POSITIONAL
        if kind == 4:
            self.kind = ParamKind.VAR_KEYWORD
        if kind in [3, 5]:
            self.kind = ParamKind.KEYWORD_ONLY
        if self.kind is None:
            self.kind = ParamKind.POSITIONAL_OR_KEYWORD
        if _type is not None:
            self.type_annotation = TypeDescriptor(_type)

    def to_proto(self) -> symbols_pb2.ParameterSymbol:
        pb_parameter = symbols_pb2.ParameterSymbol()
        pb_parameter.name = self.name
        pb_parameter.kind = symbols_pb2.ParameterKind.Value(self.kind.name)
        pb_parameter.has_default = self.has_default
        if self.type_annotation is not None:
            pb_parameter.type_annotation.CopyFrom(self.type_annotation.to_proto())
        return pb_parameter


class OverloadedFunctionSymbol:
    def __init__(self, overloaded_func_def: mpn.OverloadedFuncDef):
        self.name = overloaded_func_def.name
        self.fullname = overloaded_func_def.fullname
        self.is_final = overloaded_func_def.is_final
        self.is_property = overloaded_func_def.is_property
        self.is_static = overloaded_func_def.is_static
        self.definitions = []
        for item in overloaded_func_def.items:
            if isinstance(item, mpn.FuncDef):
                # Should not happen?
                self.definitions.append(FunctionSymbol(item))
            if isinstance(item, mpn.Decorator):
                self.definitions.append(FunctionSymbol(item.func, decorators=item.decorators))

    def to_proto(self) -> symbols_pb2.OverloadedFunctionSymbol:
        pb_overloaded_func = symbols_pb2.OverloadedFunctionSymbol()
        pb_overloaded_func.name = self.name
        pb_overloaded_func.fullname = self.fullname
        pb_overloaded_func.is_final = self.is_final
        pb_overloaded_func.is_property = self.is_property
        pb_overloaded_func.is_static = self.is_static
        for definition in self.definitions:
            pb_overloaded_func.definitions.append(definition.to_proto())
        return pb_overloaded_func


class FunctionSymbol:
    def __init__(self, func_def: mpn.FuncDef, decorators=None):
        self.name = func_def.name
        self.fullname = func_def.fullname
        self.return_type = extract_return_type(func_def)
        self.arguments = extract_arguments(func_def)
        self.has_decorators = func_def.is_decorated
        self.is_abstract = func_def.is_abstract
        self.is_async_generator = func_def.is_async_generator
        self.is_awaitable_coroutine = func_def.is_awaitable_coroutine
        self.is_coroutine = func_def.is_coroutine
        self.is_final = func_def.is_final
        self.is_overload = func_def.is_overload
        self.is_property = func_def.is_property
        self.is_static = func_def.is_static
        self.is_class_method = func_def.is_class
        self.resolved_decorator_names = []
        if self.has_decorators and decorators is not None:
            for dec in decorators:
                if isinstance(dec, mpn.NameExpr):
                    # decorator full name might not be the actual fully qualified name if it could not be resolved
                    # TODO: handle "None" case and check for fallbacks
                    if dec.fullname is not None:
                        self.resolved_decorator_names.append(dec.fullname)
            ...

    def to_proto(self) -> symbols_pb2.FunctionSymbol:
        pb_func = symbols_pb2.FunctionSymbol()
        pb_func.name = self.name
        pb_func.fully_qualified_name = self.fullname
        pb_func.has_decorators = self.has_decorators
        pb_func.resolved_decorator_names.extend(self.resolved_decorator_names)
        pb_func.is_abstract = self.is_abstract
        pb_func.is_async_generator = self.is_async_generator
        pb_func.is_awaitable_coroutine = self.is_awaitable_coroutine
        pb_func.is_coroutine = self.is_coroutine
        pb_func.is_final = self.is_final
        pb_func.is_overload = self.is_overload
        pb_func.is_property = self.is_property
        pb_func.is_static = self.is_static
        pb_func.is_class_method = self.is_class_method
        if self.return_type is not None:
            pb_func.return_annotation.CopyFrom(self.return_type.to_proto())
        for argument in self.arguments:
            pb_func.parameters.append(argument.to_proto())
        return pb_func


class ClassSymbol:
    def __init__(self, type_info: mpn.TypeInfo):
        self.name = type_info.name
        self.fullname = type_info.fullname
        self.super_classes = []
        self.mro = []
        self.methods = []
        self.overloaded_methods = []
        self.is_enum = type_info.is_enum
        self.is_generic = type_info.is_generic()
        self.is_named_tuple = type_info.is_named_tuple
        self.is_protocol = type_info.is_protocol
        self.metaclass_name = None
        self.metaclass_type = None
        for base in type_info.bases:
            if isinstance(base, mpt.Instance):
                self.super_classes.append(base.type.fullname)
        if not type_info.bad_mro and len(type_info.mro) > 2:
            for mro_type in type_info.mro:
                if (mro_type.fullname not in [b.type.fullname for b in type_info.bases]
                        and mro_type.fullname not in [type_info.fullname, "builtins.object"]):
                    # Avoid obvious elements in mro
                    self.mro.append(mro_type.fullname)
        for key in type_info.names:
            name = type_info.names.get(key)
            node = name.node
            if isinstance(node, mpn.FuncDef):
                self.methods.append(FunctionSymbol(node))
            if isinstance(node, mpn.Decorator):
                self.methods.append(FunctionSymbol(node.func, decorators=node.decorators))
            if isinstance(node, mpn.OverloadedFuncDef):
                self.overloaded_methods.append(OverloadedFunctionSymbol(node))
        class_def = type_info.defn
        self.has_metaclass = class_def.metaclass is not None
        if class_def.metaclass is not None:
            self.has_metaclass = True
            if isinstance(class_def.metaclass, mpn.NameExpr):
                self.metaclass_name = class_def.metaclass.fullname
            if isinstance(metaclass_type := type_info.metaclass_type, mpt.Instance):
                self.metaclass_type = TypeDescriptor(metaclass_type)
        self.has_decorators = len(class_def.decorators) > 0

    def to_proto(self) -> symbols_pb2.ClassSymbol:
        pb_class = symbols_pb2.ClassSymbol()
        pb_class.name = self.name
        pb_class.fully_qualified_name = self.fullname
        pb_class.super_classes.extend(self.super_classes)
        pb_class.mro.extend(self.mro)
        pb_class.has_decorators = self.has_decorators
        pb_class.has_metaclass = self.has_metaclass
        pb_class.is_enum = self.is_enum
        pb_class.is_generic = self.is_generic
        pb_class.is_named_tuple = self.is_named_tuple
        pb_class.is_protocol = self.is_protocol
        if self.metaclass_name is not None:
            pb_class.metaclass_name = self.metaclass_name
        if self.metaclass_type is not None:
            pb_class.metaclass_type.CopyFrom(self.metaclass_type.to_proto())
        for method in self.methods:
            pb_class.methods.append(method.to_proto())
        for overloaded_method in self.overloaded_methods:
            pb_class.overloaded_methods.append(overloaded_method.to_proto())
        return pb_class


class ModuleSymbol:
    def __init__(self, mypy_file: mpn.MypyFile):
        self.name = mypy_file.name
        self.fullname = mypy_file.fullname
        self.classes = []
        self.functions = []
        self.overloaded_functions = []
        for key in mypy_file.names:
            name = mypy_file.names.get(key)
            if name.fullname.startswith(mypy_file.fullname):
                symbol_table_node = name.node
                if isinstance(symbol_table_node, mpn.FuncDef):
                    self.functions.append(FunctionSymbol(symbol_table_node))
                if isinstance(symbol_table_node, mpn.OverloadedFuncDef):
                    self.overloaded_functions.append(OverloadedFunctionSymbol(symbol_table_node))
                if isinstance(symbol_table_node, mpn.TypeInfo):
                    self.classes.append(ClassSymbol(symbol_table_node))

    def to_proto(self) -> symbols_pb2.ModuleSymbol:
        pb_module = symbols_pb2.ModuleSymbol()
        pb_module.name = self.name
        pb_module.fully_qualified_name = self.fullname
        for cls in self.classes:
            pb_module.classes.append(cls.to_proto())
        for func in self.functions:
            pb_module.functions.append(func.to_proto())
        for overloaded_func in self.overloaded_functions:
            pb_module.overloaded_functions.append(overloaded_func.to_proto())
        return pb_module


def extract_arguments(func_def: mpn.FuncDef):
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


def save_module(mypy_file: mpn.MypyFile, save_as_text=True, output_dir_name="output"):
    ms = ModuleSymbol(mypy_file)
    ms_pb = ms.to_proto()
    save_dir = f"../{output_dir_name}" if save_as_text else f"../{output_dir_name}_binary"
    save_string = str(ms_pb) if save_as_text else ms_pb.SerializeToString()
    open_mode = "w" if save_as_text else "wb"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_dir}/{ms.fullname}.protobuf", open_mode) as f:
        f.write(save_string)
