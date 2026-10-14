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

from itertools import permutations

from mypy import build

from serializer.serializers import get_options, get_sources
from serializer.symbols import ModuleSymbol


def test_circular_imports_preserve_types_in_every_source_order(tmp_path):
    package = tmp_path / "probe"
    package.mkdir()
    (package / "__init__.pyi").write_text("")
    (package / "base.pyi").write_text(
        "from probe import child\n"
        "class Base:\n"
        "    def child(self) -> child.Child: ...\n"
    )
    (package / "child.pyi").write_text(
        "from probe import base\n"
        "class Child(base.Base): ...\n"
    )
    sources, _ = get_sources(str(tmp_path))
    serialized_models = []
    for source_order in permutations(sources):
        result = build.build(list(source_order), get_options())
        assert result.errors == []
        base = ModuleSymbol(result.files["probe.base"]).to_proto()
        child = ModuleSymbol(result.files["probe.child"]).to_proto()
        assert list(child.classes[0].super_classes) == ["probe.base.Base"]
        method = base.classes[0].methods[0]
        assert base.type_table[method.return_annotation_id - 1].fully_qualified_name == "probe.child.Child"
        serialized_models.append((base.SerializeToString(), child.SerializeToString()))
    assert len(set(serialized_models)) == 1
