import unittest
import pytest


class MyTest(unittest.TestCase):

    @unittest.skip  # Noncompliant {{Provide a reason for skipping this test.}}
#   ^^^^^^^^^^^^^^
    def test_unittest_skip_no_reason(self):
        self.assertEqual(1 / 0, 99)

    @unittest.skip(test)
    def test_unittest_skip_no_reason(self):
        self.assertEqual(1 / 0, 99)

    @unittest.skip("")  # Noncompliant {{Provide a reason for skipping this test.}}
#                  ^^
    def test_unittest_skip_empty_reason(self):
        self.assertEqual(1 / 0, 99)

    @unittest.skip(" ")
    def test_unittest_skip_blank_reason(self):
        self.assertEqual(1 / 0, 99)

    @bob.bob
    def test_qualified_expression_null_symbol(self):
        self.assertEqual(1 / 0, 99)

# Pytest
class UsingPytest():
    @pytest.mark.skip  # Noncompliant {{Provide a reason for skipping this test.}}
#   ^^^^^^^^^^^^^^^^^
    def test_pytest_mark_skip_no_reason():
        assert 1 == 2

    @skip()
    def test_skip_symbol_not_defined_ok():
        assert 1 == 2

    @"my string"
    def test_decorator_without_symbol_ok():
        assert 1 == 2

    @pytest.mark.skip()  # Noncompliant {{Provide a reason for skipping this test.}}
#   ^^^^^^^^^^^^^^^^^^^
    def test_skip_without_args_ko():
        assert 1 == 2

    @pytest.mark.skip("")  # Noncompliant {{Provide a reason for skipping this test.}}
#                     ^^
    def test_pytest_mark_skip_empty_reason():
        assert 1 == 2

    def test_skipped_no_reason():
        pytest.skip()  # Noncompliant {{Provide a reason for skipping this test.}}
#       ^^^^^^^^^^^^^

    def test_skipped_empty_reason():
        pytest.skip("")  # Noncompliant {{Provide a reason for skipping this test.}}
#                   ^^

    def test_skipped_unpacked_variable_ok():
        arg = [""]
        pytest.skip(*arg)

    def test_skipped_valid_with_valid_reason_in_var():
        myReason = "valid reason"
        pytest.skip(myReason)

    @["abc"]()
    def test_pytest_mark_skip_no_reason_list_literal():
        assert 1 == 2
