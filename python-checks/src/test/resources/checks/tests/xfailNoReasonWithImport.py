from pytest.mark import xfail


class UsingPytest:
    @xfail  # Noncompliant {{Provide a reason for marking this test as expected to fail.}}
#   ^^^^^^
    def test_xfail_no_reason(self):
        assert 1 == 2

    @xfail()  # Noncompliant {{Provide a reason for marking this test as expected to fail.}}
#   ^^^^^^^^
    def test_xfail_call_without_reason(self):
        assert 1 == 2

    @xfail(reason="Issue #456: known bug")
    def test_xfail_with_reason(self):
        assert 1 == 2
