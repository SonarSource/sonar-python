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
from pathlib import Path

from mypy import build

from serializer.serializers import (
    CUSTOM_STUBS_PATH,
    TYPESHED_PATH,
    get_options,
    get_sources,
)
from serializer.symbols import ModuleSymbol


def test_stdlib_dependencies_use_pinned_typeshed():
    # These public APIs are defined in the pin itself. Newer bundled Typeshed
    # moves their definitions into internal modules absent from this checkout.
    classes_by_module = {
        "io": "BytesIO",
        "email.policy": "Policy",
        "ssl": "SSLContext",
        "hashlib": "_Hash",
        "ctypes": "BigEndianStructure",
        "asyncio": "Task",
        "contextvars": "Context",
        "queue": "SimpleQueue",
        "sqlite3": "Connection",
        "struct": "Struct",
    }
    probe = build.BuildSource(
        None,
        "stdlib_probe",
        "\n".join(f"import {module}" for module in classes_by_module),
    )
    result = build.build([probe], get_options())
    stdlib = Path(TYPESHED_PATH) / "stdlib"
    for module, class_name in classes_by_module.items():
        source = result.files[module]
        assert Path(source.path).is_relative_to(stdlib), (module, source.path)
        serialized = ModuleSymbol(source).to_proto()
        assert class_name in {cls.name for cls in serialized.classes}
    # Imported stdlib dependencies must follow the same pin, including builtins
    # and helper modules such as _typeshed and typing_extensions.
    for module, source in result.files.items():
        if module != "stdlib_probe":
            assert Path(source.path).is_relative_to(stdlib), (module, source.path)


def test_custom_django_http_exports_json_response():
    custom_stubs, _ = get_sources(CUSTOM_STUBS_PATH)
    probe = build.BuildSource(
        None,
        "django_json_response_probe",
        "from django.http import JsonResponse\nresponse = JsonResponse({})",
    )

    result = build.build([*custom_stubs, probe], get_options())

    assert not any(
        'Module "django.http" has no attribute "JsonResponse"' in error
        for error in result.errors
    )
    django_http_response = ModuleSymbol(result.files["django.http.response"]).to_proto()
    assert "JsonResponse" in {cls.name for cls in django_http_response.classes}


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
        "from probe import base\nclass Child(base.Base): ...\n"
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
        assert (
            base.type_table[method.return_annotation_id - 1].fully_qualified_name
            == "probe.child.Child"
        )
        serialized_models.append((base.SerializeToString(), child.SerializeToString()))
    assert len(set(serialized_models)) == 1
