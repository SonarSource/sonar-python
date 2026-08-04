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

class PytestDefaultFixtureScopeCheckTest {

  private static final String FILE_NAME = "test_pytest_default_fixture_scope.py";
  private static final PytestDefaultFixtureScopeCheck CHECK = new PytestDefaultFixtureScopeCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/pytestDefaultFixtureScope.py", CHECK);
  }

  @Test
  void test_scope() {
    assertThat(CHECK.scope()).isEqualTo(PythonCheck.CheckScope.ALL);
  }

  @Test
  void quick_fix_message() {
    String before = """
      import pytest

      @pytest.fixture(scope="function")
      def sample():
          return 1
      """;
    PythonQuickFixVerifier.verifySemanticQuickFixMessages(CHECK, FILE_NAME, before, "Remove scope=\"function\"");
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

        @pytest.fixture(scope="function")
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

        @pytest.fixture(scope="function", autouse=True)
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture(autouse=True)
        def sample():
            return 1
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture(autouse=True, scope="function")
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture(autouse=True)
        def sample():
            return 1
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture(name="renamed", scope="function", autouse=True)
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture(name="renamed", autouse=True)
        def sample():
            return 1
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture(autouse=True, scope="function",)
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture(autouse=True)
        def sample():
            return 1
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture(
            name="renamed",
            scope="function",
            autouse=True)
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture(
            name="renamed",
            autouse=True)
        def sample():
            return 1
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture(
            scope="function",
            autouse=True)
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture(
            autouse=True)
        def sample():
            return 1
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture(
            name="renamed",
            scope="function")
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture(
            name="renamed")
        def sample():
            return 1
        """),
      Arguments.of("""
        import pytest

        @pytest.fixture(scope="function",
            autouse=True)
        def sample():
            return 1
        """, """
        import pytest

        @pytest.fixture(autouse=True)
        def sample():
            return 1
        """)
    );
  }

}
