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
import static org.assertj.core.api.Assertions.assertThatCode;

class AssertionArgumentOrderCheckTest {
  private static final AssertionArgumentOrderCheck CHECK = new AssertionArgumentOrderCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/test_assertionArgumentOrder.py", CHECK);
  }

  @Test
  void test_pytest() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/test_assertionArgumentOrderPytest.py", CHECK);
  }

  @Test
  void test_assertpy_consistent_expected_first() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/test_assertionArgumentOrderAssertpy.py", CHECK);
  }

  @Test
  void test_consistent_actual_first() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/test_assertionArgumentOrderExpectedLeft.py", CHECK);
  }

  @Test
  void test_scope() {
    assertThat(CHECK.scope()).isEqualTo(PythonCheck.CheckScope.TESTS);
  }

  @Test
  void unify_quick_fixes_unittest() {
    String before = """
      import unittest

      def value():
          return 41 + 1

      class MyTest(unittest.TestCase):
          def test_order(self):
              self.assertEqual(first=42, second=value())
              self.assertEqual(value(), 42)
      """;
    String putExpectedSecond = """
      import unittest

      def value():
          return 41 + 1

      class MyTest(unittest.TestCase):
          def test_order(self):
              self.assertEqual(first=value(), second=42)
              self.assertEqual(value(), 42)
      """;
    String putActualSecond = """
      import unittest

      def value():
          return 41 + 1

      class MyTest(unittest.TestCase):
          def test_order(self):
              self.assertEqual(first=42, second=value())
              self.assertEqual(42, value())
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "unittest_case.py", before, putExpectedSecond, putActualSecond);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "unittest_case.py", before,
      "Put all expected values second",
      "Put all actual values second");
  }

  @Test
  void unify_quick_fixes_pytest() {
    String before = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
          assert value() == 42
      """;
    String putExpectedSecond = """
      def value():
          return 41 + 1

      def test_order():
          assert value() == 42
          assert value() == 42
      """;
    String putActualSecond = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
          assert 42 == value()
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_case.py", before, putExpectedSecond, putActualSecond);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_pytest_case.py", before,
      "Put all expected values second",
      "Put all actual values second");
  }

  @Test
  void unify_quick_fixes_pytest_approx() {
    String before = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert 42 == pytest.approx(value(), abs=0.1)
          assert value() == pytest.approx(42, abs=0.1)
      """;
    String putExpectedSecond = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert value() == pytest.approx(42, abs=0.1)
          assert value() == pytest.approx(42, abs=0.1)
      """;
    String putActualSecond = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert 42 == pytest.approx(value(), abs=0.1)
          assert pytest.approx(42, abs=0.1) == value()
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_approx_case.py", before, putExpectedSecond, putActualSecond);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_pytest_approx_case.py", before,
      "Put all expected values second",
      "Put all actual values second");
  }

  @Test
  void unify_quick_fixes_assertpy() {
    String before = """
      from assertpy import assert_that

      def value():
          return 41 + 1

      def test_order():
          assert_that(42).described_as("count").is_equal_to(value())
          assert_that(value()).described_as("count").is_equal_to(42)
      """;
    String putExpectedSecond = """
      from assertpy import assert_that

      def value():
          return 41 + 1

      def test_order():
          assert_that(value()).described_as("count").is_equal_to(42)
          assert_that(value()).described_as("count").is_equal_to(42)
      """;
    String putActualSecond = """
      from assertpy import assert_that

      def value():
          return 41 + 1

      def test_order():
          assert_that(42).described_as("count").is_equal_to(value())
          assert_that(42).described_as("count").is_equal_to(value())
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_assertpy_case.py", before, putExpectedSecond, putActualSecond);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_assertpy_case.py", before,
      "Put all expected values second",
      "Put all actual values second");
  }

  @Test
  void unify_quick_fix_for_multiline_operand() {
    String before = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == (
              value()
          )
          assert value() == 42
      """;
    String putExpectedSecond = """
      def value():
          return 41 + 1

      def test_order():
          assert (
              value()
          ) == 42
          assert value() == 42
      """;
    String putActualSecond = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == (
              value()
          )
          assert 42 == value()
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_multiline_case.py", before, putExpectedSecond, putActualSecond);
  }

  @Test
  void unify_quick_fix_with_windows_line_endings() {
    String before = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
          assert value() == 42
      """.replace("\n", "\r\n");
    String putExpectedSecond = """
      def value():
          return 41 + 1

      def test_order():
          assert value() == 42
          assert value() == 42
      """.replace("\n", "\r\n");
    String putActualSecond = """
      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
          assert 42 == value()
      """.replace("\n", "\r\n");
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_windows_case.py", before, putExpectedSecond, putActualSecond);
  }

  @Test
  void unify_quick_fix_pytest_approx_with_expected_keyword() {
    String before = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert 42 == pytest.approx(expected=value(), abs=0.1)
          assert value() == pytest.approx(expected=42, abs=0.1)
      """;
    String putExpectedSecond = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert value() == pytest.approx(expected=42, abs=0.1)
          assert value() == pytest.approx(expected=42, abs=0.1)
      """;
    String putActualSecond = """
      import pytest

      def value():
          return 3.14

      def test_order():
          assert 42 == pytest.approx(expected=value(), abs=0.1)
          assert pytest.approx(expected=42, abs=0.1) == value()
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_keyword_approx_case.py", before, putExpectedSecond, putActualSecond);
  }

  @Test
  void unify_quick_fix_suppressed_when_any_assertion_cannot_be_edited() {
    String before = """
      import pytest

      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
          assert 42 == pytest.approx(other=value())
          assert value() == 42
      """;
    String putActualSecond = """
      import pytest

      def value():
          return 41 + 1

      def test_order():
          assert 42 == value()
          assert 42 == pytest.approx(other=value())
          assert 42 == value()
      """;
    // expected-first group includes an approx call without a swappable expected arg, so that unify QF is suppressed
    PythonQuickFixVerifier.verifySemantic(CHECK, "test_pytest_partial_qf_case.py", before, putActualSecond);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, "test_pytest_partial_qf_case.py", before,
      "Put all actual values second");
  }

  @Test
  void leave_file_is_noop_before_initialize() {
    AssertionArgumentOrderCheck check = new AssertionArgumentOrderCheck();
    assertThatCode(check::leaveFile).doesNotThrowAnyException();
  }

  @Test
  void convert_position_to_index_handles_crlf_and_invalid_positions() throws Exception {
    Method method = AssertionArgumentOrderCheck.class.getDeclaredMethod("convertPositionToIndex", String.class, int.class, int.class);
    method.setAccessible(true);

    String code = "a = 1\r\nb = 2\r\n";
    assertThat(method.invoke(null, code, 2, 0)).isEqualTo(7);
    assertThat(method.invoke(null, code, 2, 5)).isEqualTo(12);
    assertThat(method.invoke(null, code, 99, 0)).isEqualTo(-1);
    assertThat(method.invoke(null, code, 1, -1)).isEqualTo(-1);
    assertThat(method.invoke(null, code, 1, 50)).isEqualTo(-1);
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
}
