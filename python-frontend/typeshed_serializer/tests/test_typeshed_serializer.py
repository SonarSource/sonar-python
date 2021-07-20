from unittest.mock import Mock

from serializer import typeshed_serializer, symbols, symbols_merger


def test_build_mypy_model(typeshed_stdlib):
    assert typeshed_stdlib is not None


def test_serialize_typeshed_stdlib(typeshed_stdlib):
    typeshed_serializer.walk_typeshed_stdlib = Mock(return_value=typeshed_stdlib)
    symbols.save_module = Mock()
    typeshed_serializer.serialize_typeshed_stdlib()
    assert typeshed_serializer.walk_typeshed_stdlib.call_count == 1
    assert symbols.save_module.call_count == len(typeshed_stdlib.files)


def test_serialize_typeshed_stdlib_multiple_python_version():
    typeshed_serializer.serialize_typeshed_stdlib = Mock()
    typeshed_serializer.serialize_typeshed_stdlib_multiple_python_version()
    assert typeshed_serializer.serialize_typeshed_stdlib.call_count == len(range(5, 10))
    versions_called = set()
    for call in typeshed_serializer.serialize_typeshed_stdlib.mock_calls:
        versions_called.add(call.args[1])
    assert versions_called == {(3, 5), (3, 6), (3, 7), (3, 8), (3, 9)}


def test_save_merged_symbols():
    merged_module_symbol = symbols.MergedModuleSymbol('abc', {}, {}, {})
    symbols_merger.merge_multiple_python_versions = Mock(return_value={'abc': merged_module_symbol})
    symbols.save_module = Mock()
    typeshed_serializer.save_merged_symbols()
    assert symbols.save_module.call_count == 1
    assert symbols.save_module.mock_calls[0].args[0] == merged_module_symbol
