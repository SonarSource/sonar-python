import pytest

from serializer import typeshed_serializer


@pytest.fixture(scope="session")
def typeshed_stdlib():
    return typeshed_serializer.walk_typeshed_stdlib()
