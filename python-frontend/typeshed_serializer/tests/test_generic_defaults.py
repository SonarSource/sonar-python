#
# SonarQube Python Plugin
# Copyright (C) SonarSource Sàrl
# mailto:info AT sonarsource DOT com
#
# You can redistribute and/or modify this program under the terms of
# the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Sonar Source-Available License for more details.
#
# You should have received a copy of the Sonar Source-Available License
# along with this program; if not, see https://sonarsource.com/license/ssal/
#

import pytest
import mypy.types as mpt

from serializer.generic_info import BareTypeArgumentStatus, GenericInfo, type_argument_status

from serializer import symbols
from serializer.proto_out import symbols_pb2
from serializer.symbols_merger import merge_modules
from tests.conftest import build_modules


def test_generic_defaults(tmp_path):
    stub = tmp_path / "defaults.pyi"
    stub.write_text("""from typing import Any, Generic
from typing_extensions import TypeVar
_T = TypeVar('_T')
_Default = TypeVar('_Default', default=bool | None)
_Any = TypeVar('_Any', default=Any)
_NestedAny = TypeVar('_NestedAny', default=list[Any])
Tree = int | list["Tree"]
AnyTree = Any | list["AnyTree"]
_Recursive = TypeVar('_Recursive', default=Tree)
_RecursiveAny = TypeVar('_RecursiveAny', default=AnyTree)
class Recursive(Generic[_Recursive]): ...
class RecursiveAny(Generic[_RecursiveAny]): ...
class Required(Generic[_T]): ...
class Concrete(Generic[_Default]): ...
class AnyDefault(Generic[_Any]): ...
class NestedAnyDefault(Generic[_NestedAny]): ...
class Mixed(Generic[_T, _Default]): ...
class Plain: ...
""")
    module = symbols.ModuleSymbol(build_modules({"defaults": str(stub)})["defaults"])
    for proto in [module.to_proto(), merge_modules({"defaults"}, {"310": {"defaults": module}})["defaults"].to_proto()]:
        classes = {cls.name: cls for cls in proto.classes}
        for name in ('Concrete', 'Recursive'):
            assert classes[name].generic_info.generic == symbols_pb2.GenericInfo.NO_ANY
            assert classes[name].is_generic
        for name in ('Required', 'AnyDefault', 'NestedAnyDefault', 'Mixed', 'RecursiveAny'):
            assert classes[name].generic_info.generic == symbols_pb2.GenericInfo.CONTAINS_ANY
            assert classes[name].generic_info.WhichOneof("classification") == "generic"
        assert classes['Plain'].generic_info.WhichOneof("classification") == "non_generic"
        assert not classes['Plain'].is_generic


@pytest.mark.parametrize("default, concrete", [
    ("type[int]", True),
    ("type[Any]", False),
    ("Callable[[int], str]", True),
    ("Callable[[Any], str]", False),
    ("Callable[[int], Any]", False),
    ("Callable[..., str]", False),
    ("tuple[int, str]", True),
    ("tuple[int, Any]", False),
    ("Literal[1]", True),
])
def test_nested_generic_defaults(tmp_path, default, concrete):
    stub = tmp_path / "nested_defaults.pyi"
    stub.write_text(f"""from typing import Any, Callable, Generic, Literal
from typing_extensions import TypeVar
_Default = TypeVar('_Default', default={default})
class Defaulted(Generic[_Default]): ...
""")
    module = symbols.ModuleSymbol(build_modules({"nested_defaults": str(stub)})["nested_defaults"])
    merged = merge_modules({"nested_defaults"}, {"310": {"nested_defaults": module}})["nested_defaults"]
    for proto in (module.to_proto(), merged.to_proto()):
        defaulted = next(cls for cls in proto.classes if cls.name == "Defaulted")
        assert defaulted.is_generic
        assert defaulted.generic_info.generic == (symbols_pb2.GenericInfo.NO_ANY if concrete else symbols_pb2.GenericInfo.CONTAINS_ANY)


def test_unsupported_defaults_are_unknown(tmp_path):
    stub = tmp_path / "unknown_defaults.pyi"
    stub.write_text("""from typing import Generic, TypedDict
from typing_extensions import TypeVar
class Payload(TypedDict):
    value: int
_T = TypeVar('_T', default=Payload)
class Box(Generic[_T]): ...
""")
    module = symbols.ModuleSymbol(build_modules({"unknown_defaults": str(stub)})["unknown_defaults"])
    box = next(cls for cls in module.to_proto().classes if cls.name == "Box")
    assert box.generic_info.WhichOneof("classification") == "generic"
    assert box.generic_info.generic == symbols_pb2.GenericInfo.UNKNOWN


@pytest.mark.parametrize("parameters, expected", [
    ("_D", "UNKNOWN"),
    ("_P", "UNKNOWN"),
    ("_First, _Second", "UNKNOWN"),
    ("_D, _Any", "CONTAINS_ANY"),
    ("_Any, _D", "CONTAINS_ANY"),
])
def test_unknown_and_any_defaults(tmp_path, parameters, expected):
    stub = tmp_path / "mixed_defaults.pyi"
    stub.write_text(f"""from typing import Any, Generic, TypedDict
from typing_extensions import TypeVar, ParamSpec
class Payload(TypedDict):
    value: int
_D = TypeVar('_D', default=Payload)
_P = ParamSpec('_P', default=[int, str])
_First = TypeVar('_First', default=int)
_Second = TypeVar('_Second', default=_First)
_Any = TypeVar('_Any', default=Any)
class Box(Generic[{parameters}]): ...
""")
    module = symbols.ModuleSymbol(build_modules({"mixed_defaults": str(stub)})["mixed_defaults"])
    box = next(cls for cls in module.to_proto().classes if cls.name == "Box")
    assert box.generic_info.WhichOneof("classification") == "generic"
    assert box.generic_info.generic == symbols_pb2.GenericInfo.BareTypeArgumentStatus.Value(expected)


def test_generic_information_distinguishes_python_versions(tmp_path):
    stub = tmp_path / "versioned_defaults.pyi"
    stub.write_text("""import sys
from typing import Generic, TypedDict
from typing_extensions import TypeVar
class Payload(TypedDict):
    value: int
if sys.version_info >= (3, 11):
    _T = TypeVar('_T', default=Payload)
else:
    _T = TypeVar('_T')
class Box(Generic[_T]): ...
""")
    models = {
        str(major) + str(minor): {"versioned_defaults": symbols.ModuleSymbol(
            build_modules({"versioned_defaults": str(stub)}, (major, minor))["versioned_defaults"])}
        for major, minor in [(3, 10), (3, 11)]
    }
    merged = merge_modules({"versioned_defaults"}, models)["versioned_defaults"].to_proto()
    boxes = [cls for cls in merged.classes if cls.name == "Box"]
    assert len(boxes) == 2
    assert {tuple(cls.valid_for): cls.generic_info.generic for cls in boxes} == {
        ("310",): symbols_pb2.GenericInfo.CONTAINS_ANY,
        ("311",): symbols_pb2.GenericInfo.UNKNOWN,
    }


def test_unknown_genericity_has_no_classification():
    assert GenericInfo.UNKNOWN.to_proto().WhichOneof("classification") is None


@pytest.mark.parametrize("origin", [mpt.TypeOfAny.from_unimported_type, mpt.TypeOfAny.from_error])
def test_unresolved_any_is_not_a_known_any_default(origin):
    unresolved = mpt.AnyType(origin)
    propagated = mpt.AnyType(mpt.TypeOfAny.from_another_any, source_any=unresolved)
    assert type_argument_status(unresolved) == BareTypeArgumentStatus.UNKNOWN
    assert type_argument_status(propagated) == BareTypeArgumentStatus.UNKNOWN


def test_unresolved_alias_is_unknown():
    assert type_argument_status(mpt.TypeAliasType(None, [])) == BareTypeArgumentStatus.UNKNOWN
