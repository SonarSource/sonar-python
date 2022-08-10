from pytest.mark import skip

# Pytest
class UsingPytest():
    @skip()  # Noncompliant {{Provide a reason for skipping this test.}}
#   ^^^^^^^
    def test_pytest_mark_skip_no_reason():
        assert 1 == 2

    @skip  # Noncompliant {{Provide a reason for skipping this test.}}
#   ^^^^^
    def test_pytest_mark_skip_name():
        assert 1 == 2
