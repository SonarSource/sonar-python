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

from serializer import symbols
from serializer.proto_out import symbols_pb2
from tests.conftest import build_modules


def test_bound_callable_aliases_preserve_signatures(tmp_path):
    stub = tmp_path / "aliases.pyi"
    stub.write_text("""from typing import Callable, overload
from typing_extensions import Self
class Factory:
    @classmethod
    def create(cls, name: str, /, *, enabled: bool = ...) -> Self: ...
    @classmethod
    @overload
    def open(cls, name: str) -> Self: ...
    @classmethod
    @overload
    def open(cls, name: bytes, *, encoding: str) -> Self: ...
create = Factory.create
open = Factory.open
callback: Callable[[int], str]
class_alias = Factory
""")
    module = symbols.ModuleSymbol(build_modules({"aliases": str(stub)})["aliases"])
    functions = {f.name: f.to_proto() for f in module.functions}
    create = functions["create"]
    assert [p.name for p in create.parameters] == ["name", "enabled"]
    assert create.parameters[0].kind == symbols_pb2.POSITIONAL_ONLY
    assert create.parameters[1].kind == symbols_pb2.KEYWORD_ONLY
    assert create.parameters[1].has_default
    assert not create.has_decorators
    assert not create.is_class_method
    assert create.return_annotation.fully_qualified_name == "aliases.Factory"
    overloaded = next(f for f in module.overloaded_functions if f.name == "open")
    assert len(overloaded.definitions) == 2
    assert all(f.return_type.fully_qualified_name == "aliases.Factory" for f in overloaded.definitions)
    assert [[p.name for p in f.parameters] for f in overloaded.definitions] == [["name"], ["name", "encoding"]]
    assert "callback" in {v.name for v in module.vars}
    assert "class_alias" not in functions


def test_callable_aliases_keep_positional_only_names(tmp_path):
    stub = tmp_path / "names.pyi"
    stub.write_text("""from typing import overload
def function(value: int, /, optional: str = ...) -> int: ...
def legacy(__value: int) -> int: ...
def no_arguments() -> int: ...
@overload
def overloaded(value: int, /) -> int: ...
@overload
def overloaded(value: str, /) -> str: ...
class Factory:
    def method(self, value: int, /) -> int: ...
    @staticmethod
    def static(value: int, /) -> int: ...
    @classmethod
    def empty(cls) -> int: ...
factory: Factory
function_alias = function
legacy_alias = legacy
empty_alias = no_arguments
overloaded_alias = overloaded
unbound_alias = Factory.method
bound_alias = factory.method
static_alias = Factory.static
empty_method_alias = Factory.empty
""")
    module = symbols.ModuleSymbol(build_modules({"names": str(stub)})["names"])
    functions = {f.name: f.to_proto() for f in module.functions}
    expected = {
        "function_alias": ["value", "optional"],
        "legacy_alias": ["__value"],
        "empty_alias": [],
        "unbound_alias": ["self", "value"],
        "bound_alias": ["value"],
        "static_alias": ["value"],
        "empty_method_alias": [],
    }
    for name, names in expected.items():
        assert [p.name for p in functions[name].parameters] == names
        if names:
            assert functions[name].parameters[0].kind == symbols_pb2.POSITIONAL_ONLY
    assert functions["function_alias"].parameters[1].has_default
    overloaded = next(f for f in module.overloaded_functions if f.name == "overloaded_alias")
    assert [[p.name for p in f.parameters] for f in overloaded.definitions] == [["value"], ["value"]]


@pytest.mark.parametrize("signature, names, kinds", [
    ("left: int, right: str", ["left", "right"], [symbols_pb2.POSITIONAL_OR_KEYWORD] * 2),
    ("value: int", ["value"], [symbols_pb2.POSITIONAL_OR_KEYWORD]),
    ("a: int, b: int, c: int", ["a", "b", "c"], [symbols_pb2.POSITIONAL_OR_KEYWORD] * 3),
    ("left: int, *, right: str", ["left", "right"],
     [symbols_pb2.POSITIONAL_OR_KEYWORD, symbols_pb2.KEYWORD_ONLY]),
    ("left: int, right: str, /", ["", ""], [symbols_pb2.POSITIONAL_ONLY] * 2),
])
def test_callable_aliases_preserve_specialized_paramspec_names(tmp_path, signature, names, kinds):
    stub = tmp_path / "specialized.pyi"
    stub.write_text(f"""from typing import Callable, Generic, TypeVar
from typing_extensions import ParamSpec
P = ParamSpec("P")
R = TypeVar("R")
class Wrapper(Generic[P, R]):
    def __init__(self, f: Callable[P, R]) -> None: ...
    def call(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
def target({signature}) -> bool: ...
wrapped = Wrapper(target)
alias = wrapped.call
""")
    module = symbols.ModuleSymbol(build_modules({"specialized": str(stub)})["specialized"])
    alias = next(f for f in module.functions if f.name == "alias").to_proto()
    assert [p.name for p in alias.parameters] == names
    assert [p.kind for p in alias.parameters] == kinds
    assert alias.return_annotation.fully_qualified_name == "builtins.bool"
    assert not alias.has_decorators


@pytest.mark.parametrize("signature, expected_name, expected_kind", [
    ('Arg(int, "actual")', "actual", symbols_pb2.POSITIONAL_OR_KEYWORD),
    ("int", "", symbols_pb2.POSITIONAL_ONLY),
])
def test_callable_aliases_preserve_decorator_signature_names(tmp_path, signature, expected_name, expected_kind):
    stub = tmp_path / "decorated.pyi"
    stub.write_text(f"""from typing import Callable
from mypy_extensions import Arg
def transform(f: Callable[..., int]) -> Callable[[{signature}], int]: ...
@transform
def original(old: int, removed: str) -> int: ...
alias = original
""")
    module = symbols.ModuleSymbol(build_modules({"decorated": str(stub)})["decorated"])
    alias = next(f for f in module.functions if f.name == "alias").to_proto()
    assert [p.name for p in alias.parameters] == [expected_name]
    assert alias.parameters[0].kind == expected_kind
    assert alias.has_decorators
    assert list(alias.resolved_decorator_names) == ["decorated.transform"]


def test_callable_aliases_only_keep_async_flag_for_async_return_types(tmp_path):
    stub = tmp_path / "asynchronous.pyi"
    stub.write_text("""from typing import Any, AsyncIterator, Callable, Coroutine, Generic, TypeVar
T = TypeVar("T")
def sync(f: Callable[..., Coroutine[Any, Any, str]]) -> Callable[[int], str]: ...
def identity(f: Callable[[int], T]) -> Callable[[int], T]: ...
@sync
async def transformed(value: int) -> str: ...
@identity
async def preserved(value: int) -> str: ...
async def plain(value: int) -> str: ...
async def generator() -> AsyncIterator[str]:
    yield "value"
class Box(Generic[T]):
    async def get(self) -> T: ...
box: Box[str]
sync_alias = transformed
preserved_alias = preserved
plain_alias = plain
generator_alias = generator
bound_alias = box.get
""")
    module = symbols.ModuleSymbol(build_modules({"asynchronous": str(stub)})["asynchronous"])
    aliases = {f.name: f.to_proto() for f in module.functions if f.name.endswith("_alias")}
    assert {name: alias.is_asynchronous for name, alias in aliases.items()} == {
        "sync_alias": False,
        "preserved_alias": True,
        "plain_alias": True,
        "generator_alias": True,
        "bound_alias": True,
    }
    assert aliases["sync_alias"].return_annotation.fully_qualified_name == "builtins.str"
    assert aliases["sync_alias"].has_decorators
    assert list(aliases["sync_alias"].resolved_decorator_names) == ["asynchronous.sync"]
    assert aliases["bound_alias"].return_annotation.args[-1].fully_qualified_name == "builtins.str"
