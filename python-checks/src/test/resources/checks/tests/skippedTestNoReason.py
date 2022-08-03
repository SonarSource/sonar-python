import unittest
import pytest


class MyTest(unittest.TestCase):

    @unittest.skip  # Noncompliant {{Provide a reason for skipping this test.}}
#   ^^^^^^^^^^^^^^
    def test_unittest_skip_no_reason(self):
        self.assertEqual(1 / 0, 99)

    @unittest.skip("")  # Noncompliant {{Provide a reason for skipping this test.}}
#                  ^^
    def test_unittest_skip_empty_reason(self):
        self.assertEqual(1 / 0, 99)

    @unittest.skip(" ")
    def test_unittest_skip_blank_reason(self):
        self.assertEqual(1 / 0, 99)


# Pytest
class UsingPytest():
    @pytest.mark.skip  # Noncompliant {{Provide a reason for skipping this test.}}
#   ^^^^^^^^^^^^^^^^^
    def test_pytest_mark_skip_no_reason():
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
