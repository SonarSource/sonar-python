import os


def test_suffix():
    os.environ['API_KEY'] = 'test_key'  # Noncompliant {{Use the "monkeypatch" fixture for temporary modifications instead of manually modifying global state.}}
