import os
from unittest.mock import Mock

import pytest
from mypy import build

from serializer import typeshed_serializer


@pytest.fixture(scope="session")
def typeshed_stdlib():
    return typeshed_serializer.walk_typeshed_stdlib()


@pytest.fixture(scope="session")
def fake_module_36_38():
    fake_module_path = os.path.join(os.path.dirname(__file__), "resources/fakemodule.pyi")
    typeshed_serializer.load_single_module = Mock(return_value=build.BuildSource(fake_module_path, "fakemodule"))
    fake_module_36 = typeshed_serializer.build_single_module('fakemodule', python_version=(3, 6))
    fake_module_38 = typeshed_serializer.build_single_module('fakemodule', python_version=(3, 8))
    return [fake_module_36, fake_module_38]
