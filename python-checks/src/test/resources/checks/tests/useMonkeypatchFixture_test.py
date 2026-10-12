import os


def test_suffix():
    os.environ['API_KEY'] = 'test_key'  # Noncompliant {{Use the "monkeypatch" fixture to modify this global state, otherwise it must be restored manually.}}
