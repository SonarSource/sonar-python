# Not a pytest-collected file name — no issues expected.


def test_file_operation(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('data')
    assert temp_dir.exists()
