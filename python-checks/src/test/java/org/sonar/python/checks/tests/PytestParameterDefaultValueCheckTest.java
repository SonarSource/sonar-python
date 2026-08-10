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

class PytestParameterDefaultValueCheckTest {

  private static final String FILE_NAME = "test_pytest_parameter_default_value.py";
  private static final PytestParameterDefaultValueCheck CHECK = new PytestParameterDefaultValueCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/test_pytestParameterDefaultValue.py", CHECK);
  }

  @Test
  void test_fixtures_in_conftest() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/conftest_pytestParameterDefaultValue.py", CHECK);
  }

  @Test
  void test_no_issue_in_non_pytest_file() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/pytestParameterDefaultValueNonPytestFile.py", CHECK);
  }

  @Test
  void test_scope() {
    assertThat(CHECK.scope()).isEqualTo(PythonCheck.CheckScope.ALL);
  }

  @Test
  void quick_fix_message() {
    String before = """
      import pytest

      @pytest.fixture
      def value():
          return 42

      def test_with_default(value=1):
          assert value > 0
      """;
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, FILE_NAME, before, "Remove default value");
  }

  @ParameterizedTest
  @MethodSource("quickFixCases")
  void quick_fix(String before, String after) {
    PythonQuickFixVerifier.verifySemantic(CHECK, FILE_NAME, before, after);
  }

  static Stream<Arguments> quickFixCases() {
    return Stream.of(
      Arguments.of("""
        import pytest

        @pytest.fixture
        def value():
            return 42

        def test_with_default(value=1):
            assert value > 0
        """, """
        import pytest

        @pytest.fixture
        def value():
            return 42

        def test_with_default(value):
            assert value > 0
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture
        def value():
            return 42

        def test_typed_default(value: int = 1):
            assert value > 0
        """, """
        import pytest

        @pytest.fixture
        def value():
            return 42

        def test_typed_default(value: int):
            assert value > 0
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture
        def dep():
            return 0

        @pytest.fixture
        def fixture_with_default(dep=1):
            return dep
        """, """
        import pytest

        @pytest.fixture
        def dep():
            return 0

        @pytest.fixture
        def fixture_with_default(dep):
            return dep
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture
        def value():
            return 42

        def test_none_default(value=None):
            assert value is None
        """, """
        import pytest

        @pytest.fixture
        def value():
            return 42

        def test_none_default(value):
            assert value is None
        """)
    );
  }
}
