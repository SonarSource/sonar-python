pytest_plugins = ["pytest_cov", "pytest_timeout"]  # Noncompliant {{"pytest_plugins" should be defined in conftest.py files}}
pytest_plugins: list[str] = ["pytest_cov"]  # Noncompliant {{"pytest_plugins" should be defined in conftest.py files}}
pytest_plugins: list[str]

other_plugins = ["pytest_cov"]

def test_feature():
    assert other_plugins[0] == "pytest_cov"

def nested():
    pytest_plugins = ["pytest_cov"]

class TestClass:
    pytest_plugins = ["pytest_cov"]
