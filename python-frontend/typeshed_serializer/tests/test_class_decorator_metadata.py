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
from serializer.proto_out.symbols_pb2 import DecoratorInfo, DecoratorTypeReference
from serializer.symbols_merger import merge_modules
from tests.conftest import build_modules


def direct_decorator(name, *, is_instance=False):
    return DecoratorInfo(expression_types=[DecoratorTypeReference(
        fully_qualified_name=name, is_instance=is_instance)])


def test_markers_and_stacked_custom_decorators_survive_inline_and_merged_serialization(tmp_path):
    stub = tmp_path / "markers.pyi"
    stub.write_text("""from typing import final as final_marker
from typing import Any

def custom(cls: Any) -> Any: ...
def disjoint_marker(cls: Any) -> Any: ...
@final_marker
class Final: ...
@disjoint_marker
class Disjoint: ...
@custom
@final_marker
class Mixed: ...
@final_marker()
class Called: ...
""")
    model = build_modules({"markers": str(stub)})["markers"]
    module = symbols.ModuleSymbol(model)
    classes = {cls.name: cls.to_proto() for cls in module.classes}
    assert all(cls.has_decorators for cls in classes.values())
    assert classes["Final"].decorators == [direct_decorator("typing.final")]
    assert classes["Disjoint"].decorators == [direct_decorator("markers.disjoint_marker")]
    assert classes["Mixed"].decorators == [direct_decorator("markers.custom"), direct_decorator("typing.final")]
    assert classes["Called"].decorators == [DecoratorInfo(
        expression_types=[DecoratorTypeReference()], has_callee=True,
        callee_types=[DecoratorTypeReference(fully_qualified_name="typing.final")])]
    merged = merge_modules({"markers"}, {"313": {"markers": module}, "314": {"markers": module}})
    merged_classes = {cls.name: cls for cls in merged["markers"].to_proto().classes}
    for name, cls in classes.items():
        assert merged_classes[name].decorators == cls.decorators


def test_canonical_direct_marker_identity_survives_stub_reexports(tmp_path):
    helper = tmp_path / "helpers.pyi"
    helper.write_text("from typing import final as final\n")
    consumer = tmp_path / "consumer.pyi"
    consumer.write_text("""import helpers
from helpers import final as marker
@helpers.final
class Qualified: ...
@marker
class Direct: ...
""")
    model = build_modules({"helpers": str(helper), "consumer": str(consumer)})["consumer"]
    classes = symbols.ModuleSymbol(model).to_proto().classes
    assert len(classes) == 2
    assert all(cls.has_decorators for cls in classes)
    assert all(cls.decorators == [direct_decorator("typing.final")] for cls in classes)


def test_different_decorator_identities_are_not_merged_across_versions(tmp_path):
    stub = tmp_path / "versioned.pyi"
    stub.write_text("""import sys
from typing import final
from typing import Any
def other_marker(cls: Any) -> Any: ...
if sys.version_info >= (3, 14):
    @other_marker
    class Marked: ...
else:
    @final
    class Marked: ...
""")
    models = {
        version: {"versioned": symbols.ModuleSymbol(build_modules(
            {"versioned": str(stub)}, python_version=target)["versioned"])}
        for version, target in [("313", (3, 13)), ("314", (3, 14))]
    }
    merged = merge_modules({"versioned"}, models)["versioned"].to_proto()
    assert len(merged.classes) == 2
    by_version = {tuple(cls.valid_for): list(cls.decorators) for cls in merged.classes}
    assert by_version == {
        ("313",): [direct_decorator("typing.final")],
        ("314",): [direct_decorator("versioned.other_marker")],
    }


def test_factories_instances_and_unknowns_preserve_separate_entries(tmp_path):
    stub = tmp_path / "decorators.pyi"
    stub.write_text("""from typing import Any, Callable, final
class Marker:
    def __call__(self, cls: Any) -> Any: ...
marker: Marker
def factory() -> Callable[[Any], Any]: ...
@final
@factory()
@marker
@missing()
class Marked: ...
""")
    cls = symbols.ModuleSymbol(build_modules({"decorators": str(stub)})["decorators"]).to_proto().classes[-1]
    assert cls.decorators == [
        direct_decorator("typing.final"),
        DecoratorInfo(expression_types=[DecoratorTypeReference()], has_callee=True,
                      callee_types=[DecoratorTypeReference(fully_qualified_name="decorators.factory")]),
        direct_decorator("decorators.Marker", is_instance=True),
        DecoratorInfo(expression_types=[DecoratorTypeReference()], has_callee=True,
                      callee_types=[DecoratorTypeReference()]),
    ]
