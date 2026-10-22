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
        for definition in functions["decorated"].definitions:
            assert definition.has_decorators
            assert definition.resolved_decorator_names == ["typing.overload", ""]
