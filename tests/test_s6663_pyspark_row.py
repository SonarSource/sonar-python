"""
Tests for S6663 false positive fix: pyspark.sql.Row subscript access.

These tests verify that:
1. The test resource file (indexMethod.py) contains test cases for pyspark.sql.Row
   subscript access that should NOT be flagged as Noncompliant by rule S6663.
2. The IndexMethodCheck.java rule implementation handles Sequence subclasses
   that override __getitem__ to accept non-integer keys (e.g., str).
"""
import os
import re
import unittest


SONAR_PYTHON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESOURCE = os.path.join(
    SONAR_PYTHON_ROOT,
    "python-checks", "src", "test", "resources", "checks", "indexMethod.py",
)
RULE_IMPL = os.path.join(
    SONAR_PYTHON_ROOT,
    "python-checks", "src", "main", "java", "org", "sonar", "python", "checks",
    "IndexMethodCheck.java",
)


class TestIndexMethodPysparkRow(unittest.TestCase):
    """Verify the Python test resource file contains pyspark.sql.Row test cases."""

    def test_resource_has_pyspark_row_import(self):
        with open(TEST_RESOURCE, encoding="utf-8") as f:
            content = f.read()
        self.assertIn(
            "from pyspark.sql import Row",
            content,
            "indexMethod.py must import pyspark.sql.Row for the false-positive test case",
        )

    def test_resource_has_row_string_access_without_noncompliant(self):
        with open(TEST_RESOURCE, encoding="utf-8") as f:
            lines = f.readlines()
        row_access_lines = [
            line for line in lines
            if re.search(r'row\[.*"', line, re.IGNORECASE) and "Noncompliant" not in line
        ]
        self.assertTrue(
            len(row_access_lines) > 0,
            "indexMethod.py must have at least one pyspark.sql.Row string subscript "
            "access line that is NOT marked as Noncompliant",
        )


class TestIndexMethodCheckHandlesOverriddenGetitem(unittest.TestCase):
    """Verify the Java rule implementation handles overridden __getitem__."""

    def test_rule_checks_getitem_parameter_type(self):
        with open(RULE_IMPL, encoding="utf-8") as f:
            content = f.read()
        self.assertTrue(
            "resolveMember" in content and "__getitem__" in content,
            "IndexMethodCheck.java must resolve __getitem__ member to check "
            "parameter types before flagging non-__index__ indices",
        )


if __name__ == "__main__":
    unittest.main()
