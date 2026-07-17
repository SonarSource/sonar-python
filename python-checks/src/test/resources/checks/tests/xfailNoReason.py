import pytest


class UsingPytest:
    @pytest.mark.xfail  # Noncompliant {{Provide a reason for marking this test as expected to fail.}}
#   ^^^^^^^^^^^^^^^^^^
    def test_xfail_no_reason(self):
        assert 1 == 2

    @pytest.mark.xfail()  # Noncompliant {{Provide a reason for marking this test as expected to fail.}}
#   ^^^^^^^^^^^^^^^^^^^^
    def test_xfail_call_without_reason(self):
        assert 1 == 2

    @pytest.mark.xfail(reason="Issue #456: known bug")
    def test_xfail_with_reason(self):
        assert 1 == 2

    @pytest.mark.xfail(reason="")  # Noncompliant {{Provide a reason for marking this test as expected to fail.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def test_xfail_empty_reason(self):
        assert 1 == 2

    @pytest.mark.xfail(reason=" ")
    def test_xfail_blank_reason(self):
        assert 1 == 2

    @pytest.mark.xfail(condition=True)  # Noncompliant {{Provide a reason for marking this test as expected to fail.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    def test_xfail_with_condition_only(self):
        assert 1 == 2

    @pytest.mark.xfail(reason=other_var)
    def test_xfail_with_variable_reason(self):
        assert 1 == 2

    @pytest.mark.skip
    def test_skip_not_xfail(self):
        assert 1 == 2

    @bob.bob
    def test_qualified_expression_null_symbol(self):
        assert 1 == 2

    @xfail()
    def test_xfail_symbol_not_defined_ok(self):
        assert 1 == 2

    @"my string"
    def test_decorator_without_symbol_ok(self):
        assert 1 == 2
