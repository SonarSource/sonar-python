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

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class PytestDecoratorParenthesesCheckTest {

  private static final String FILE_NAME = "test_pytest_decorator_parentheses.py";
  private static final PytestDecoratorParenthesesCheck CHECK = new PytestDecoratorParenthesesCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/pytestDecoratorParentheses.py", CHECK);
  }

  @Test
  void test_require_parentheses() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/pytestDecoratorParenthesesRequired.py", checkRequiringParentheses());
  }

  @Test
  void test_scope() {
    assertThat(CHECK.scope()).isEqualTo(PythonCheck.CheckScope.ALL);
  }

  @Test
  void quick_fix_removes_empty_parentheses() {
    String before = """
      import pytest

      @pytest.fixture()
      def sample():
          return 1
      """;
    String after = """
      import pytest

      @pytest.fixture
      def sample():
          return 1
      """;
    PythonQuickFixVerifier.verifySemantic(CHECK, FILE_NAME, before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, FILE_NAME, before, "Remove the empty parentheses");
  }

  @ParameterizedTest
  @MethodSource("removalQuickFixCases")
  void quick_fix_removes_empty_parentheses_of(String before, String after) {
    PythonQuickFixVerifier.verifySemantic(CHECK, FILE_NAME, before, after);
  }

  static Stream<Arguments> removalQuickFixCases() {
    return Stream.of(
      Arguments.of("""
        import pytest

        @pytest.mark.slow()
        def test_sample():
            assert True
        """, """
        import pytest

        @pytest.mark.slow
        def test_sample():
            assert True
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture ()
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture
        def sample():
            return 1
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture(
        )
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture
        def sample():
            return 1
        """));
  }

  @Test
  void no_quick_fix_when_a_comment_sits_inside_the_parentheses() {
    String before = """
      import pytest

      @pytest.fixture(
          # TODO decide on a scope
      )
      def sample():
          return 1
      """;
    PythonQuickFixVerifier.verifySemanticNoQuickFixes(CHECK, FILE_NAME, before);
  }

  @Test
  void quick_fix_adds_empty_parentheses() {
    String before = """
      import pytest

      @pytest.fixture
      def sample():
          return 1
      """;
    String after = """
      import pytest

      @pytest.fixture()
      def sample():
          return 1
      """;
    PythonQuickFixVerifier.verifySemantic(checkRequiringParentheses(), FILE_NAME, before, after);
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(checkRequiringParentheses(), FILE_NAME, before, "Add empty parentheses");
  }

  @Test
  void quick_fix_adds_empty_parentheses_to_mark() {
    String before = """
      import pytest

      @pytest.mark.slow
      def test_sample():
          assert True
      """;
    String after = """
      import pytest

      @pytest.mark.slow()
      def test_sample():
          assert True
      """;
    PythonQuickFixVerifier.verifySemantic(checkRequiringParentheses(), FILE_NAME, before, after);
  }

  private static PytestDecoratorParenthesesCheck checkRequiringParentheses() {
    PytestDecoratorParenthesesCheck check = new PytestDecoratorParenthesesCheck();
    check.requireParentheses = true;
    return check;
  }
}
