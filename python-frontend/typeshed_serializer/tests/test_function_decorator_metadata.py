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

from serializer import symbols
from serializer.symbols_merger import merge_modules
from tests.conftest import build_modules


def test_overloads_preserve_unknown_decorators_through_serialization_and_merge(tmp_path):
    stub = tmp_path / "decorated.pyi"
    stub.write_text("""from typing import Any, Callable, overload as marker

def factory() -> Callable[..., Any]: ...
@marker
def plain(value: int) -> int: ...
@marker
def plain(value: str) -> str: ...
@marker
@factory()
def decorated(value: int) -> int: ...
@marker
@missing()
def decorated(value: str) -> str: ...
""")
    module = symbols.ModuleSymbol(build_modules({"decorated": str(stub)})["decorated"])
    merged = merge_modules({"decorated"}, {"313": {"decorated": module}, "314": {"decorated": module}})
    for serialized in [module.to_proto(), merged["decorated"].to_proto()]:
        functions = {function.name: function for function in serialized.overloaded_functions}
        assert len(functions["plain"].definitions) == 2
        assert len(functions["decorated"].definitions) == 2
        for definition in functions["plain"].definitions:
            assert definition.resolved_decorator_names == ["typing.overload"]
        for definition, factory in zip(functions["decorated"].definitions, ["decorated.factory", ""]):
            assert definition.has_decorators
            assert definition.resolved_decorator_names == ["typing.overload", factory]


def test_deprecation_decorator_factories_preserve_identity_and_unknown_stacks(tmp_path):
    stub = tmp_path / "deprecated_api.pyi"
    stub.write_text('''from typing_extensions import deprecated as obsolete
from typing import Any, Callable, overload

def unknown_factory() -> Callable[..., Any]: ...

@obsolete("use the replacement")
def old(value: int) -> None: ...

@obsolete("use the replacement")
@unknown_factory()
def wrapped(value: int) -> None: ...

@overload
@obsolete("use the replacement")
def overloaded(value: int) -> int: ...
@overload
@obsolete("use the replacement")
def overloaded(value: str) -> str: ...
''')
    model = symbols.ModuleSymbol(build_modules({"deprecated_api": str(stub)}, python_version=(3, 10))["deprecated_api"])
    merged = merge_modules({"deprecated_api"}, {"313": {"deprecated_api": model}, "314": {"deprecated_api": model}})
    for serialized in [model.to_proto(), merged["deprecated_api"].to_proto()]:
        functions = {function.name: function for function in serialized.functions}
        assert functions["old"].has_decorators
        assert list(functions["old"].resolved_decorator_names) == ["typing_extensions.deprecated"]
        assert list(functions["wrapped"].resolved_decorator_names) == ["typing_extensions.deprecated", "deprecated_api.unknown_factory"]
        for definition in serialized.overloaded_functions[0].definitions:
            assert list(definition.resolved_decorator_names) == ["typing.overload", "typing_extensions.deprecated"]
