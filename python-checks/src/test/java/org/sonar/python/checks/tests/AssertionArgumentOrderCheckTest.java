/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.tests;

import java.lang.reflect.Method;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class AssertionArgumentOrderCheckTest {
  private static final AssertionArgumentOrderCheck CHECK = new AssertionArgumentOrderCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/test_assertionArgumentOrder.py", CHECK);
  }

  @Test
  void test_expected_on_left() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/test_assertionArgumentOrderExpectedLeft.py", checkWithExpectedOnRight());
  }

  @Test
  void test_scope() {
    assertThat(CHECK.scope()).isEqualTo(PythonCheck.CheckScope.TESTS);
  }

  @Test
  void unittest_quick_fix() {
    String before = """
      import unittest

      def value():
          return 41 + 1

      class MyTest(unittest.TestCase):
          def test_order(self):
              self.assertEqual(first=42, second=value())
      """;
    String after = """
      import unittest

      def value():
          return 41 + 1

      class MyTest(unittest.TestCase):
          def test_order(self):
              self.assertEqual(first=value(), second=42)
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "unittest_case.py", before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "unittest_case.py", before, "Swap the actual and expected arguments");
  }

  @Test
  void pytest_quick_fix() {
    String before = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
      """;
    String after = """
      def value():
          return 41 + 1

      def test_order():
          assert value() == 42
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_case.py", before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_pytest_case.py", before, "Swap the actual and expected operands");
  }

  @Test
  void pytest_approx_quick_fix() {
    String before = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert 42 == pytest.approx(value(), abs=0.1)
      """;
    String after = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert value() == pytest.approx(42, abs=0.1)
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_approx_case.py", before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_pytest_approx_case.py", before, "Swap the actual and expected operands");
  }

  @Test
  void pytest_quick_fix_expected_on_left() {
    String before = """
      def value():
          return 41 + 1

      def test_order():
          assert value() == 42
      """;
    String after = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
      """;
    PythonQuickFixVerifier.verifySemantic(checkWithExpectedOnRight(), "test_pytest_expected_left_case.py", before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(checkWithExpectedOnRight(), "test_pytest_expected_left_case.py", before,
      "Swap the actual and expected operands");
  }

  @Test
  void pytest_approx_quick_fix_expected_on_left() {
    String before = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert pytest.approx(value(), abs=0.1) == 42
      """;
    String after = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert pytest.approx(42, abs=0.1) == value()
      """;
    PythonQuickFixVerifier.verifySemantic(checkWithExpectedOnRight(), "test_pytest_approx_expected_left_case.py", before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(checkWithExpectedOnRight(), "test_pytest_approx_expected_left_case.py", before,
      "Swap the actual and expected operands");
  }

  @Test
  void assertpy_quick_fix() {
    String before = """
      from assertpy import assert_that

      def value():
          return 41 + 1

      def test_order():
          assert_that(42).described_as("count").is_equal_to(value())
      """;
    String after = """
      from assertpy import assert_that

      def value():
          return 41 + 1

      def test_order():
          assert_that(value()).described_as("count").is_equal_to(42)
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_assertpy_case.py", before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_assertpy_case.py", before, "Swap the actual and expected values");
  }

  @Test
  void quick_fix_for_multiline_operand() {
    String before = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == (
              value()
          )
      """;
    String after = """
      def value():
          return 41 + 1

      def test_order():
          assert (
              value()
          ) == 42
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_multiline_case.py", before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_pytest_multiline_case.py", before, "Swap the actual and expected operands");
  }

  @Test
  void pytest_quick_fix_with_windows_line_endings() {
    String before = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
      """.replace("\n", "\r\n");
    String after = """
      def value():
          return 41 + 1

      def test_order():
          assert value() == 42
      """.replace("\n", "\r\n");
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_windows_case.py", before, after);
  }

  @Test
  void pytest_approx_quick_fix_with_expected_keyword() {
    String before = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert 42 == pytest.approx(expected=value(), abs=0.1)
      """;
    String after = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert value() == pytest.approx(expected=42, abs=0.1)
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_keyword_approx_case.py", before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_pytest_keyword_approx_case.py", before, "Swap the actual and expected operands");
  }

  @Test
  void convert_position_to_index_handles_crlf_and_invalid_positions() throws Exception {
    Method method = AssertionArgumentOrderCheck.class.getDeclaredMethod("convertPositionToIndex", String.class, int.class, int.class);
    method.setAccessible(true);

    String code = "a = 1\r\nb = 2\r\n";
    assertThat(method.invoke(null, code, 2, 0)).isEqualTo(7);
    assertThat(method.invoke(null, code, 2, 5)).isEqualTo(12);
    assertThat(method.invoke(null, code, 4, 0)).isEqualTo(-1);
    assertThat(method.invoke(null, code, 2, -1)).isEqualTo(-1);
    assertThat(method.invoke(null, code, 2, 6)).isEqualTo(-1);
  }

  @Test
  void next_index_handles_all_line_break_variants() throws Exception {
    Method method = AssertionArgumentOrderCheck.class.getDeclaredMethod("nextIndex", String.class, int.class);
    method.setAccessible(true);

    assertThat(method.invoke(null, "a", 0)).isEqualTo(1);
    assertThat(method.invoke(null, "\n", 0)).isEqualTo(1);
    assertThat(method.invoke(null, "\r", 0)).isEqualTo(1);
    assertThat(method.invoke(null, "\r\n", 0)).isEqualTo(2);
  }

  private static AssertionArgumentOrderCheck checkWithExpectedOnRight() {
    AssertionArgumentOrderCheck check = new AssertionArgumentOrderCheck();
    check.expectedOnRight = false;
    return check;
  }
}
