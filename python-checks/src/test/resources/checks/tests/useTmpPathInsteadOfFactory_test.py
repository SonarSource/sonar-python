def test_file_operation(tmp_path_factory):  # Noncompliant {{Use "tmp_path" instead of "tmp_path_factory" in function-scoped tests.}}
#                       ^^^^^^^^^^^^^^^^
    temp_dir = tmp_path_factory.mktemp('data')
    assert temp_dir.exists()
