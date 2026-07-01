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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class TestCasesShouldContainTestsCheckTest {

  @Test
  void scope() {
    assertThat(new TestCasesShouldContainTestsCheck().scope()).isEqualTo(PythonCheck.CheckScope.TESTS);
  }

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/testCasesShouldContainTests_test.py", new TestCasesShouldContainTestsCheck());
  }

  @Test
  void pytest_file_without_tests() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/emptyPytestModule_test.py", new TestCasesShouldContainTestsCheck());
  }

  @Test
  void pytest_prefix_file_without_tests() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/test_emptyPytestModule.py", new TestCasesShouldContainTestsCheck());
  }

  @Test
  void non_pytest_file() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/helpersOnlyModule.py", new TestCasesShouldContainTestsCheck());
  }

  @Test
  void empty_file() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/emptyModule.py", new TestCasesShouldContainTestsCheck());
  }

  @Test
  void file_with_class_tests_but_no_candidate_class() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/test_fileWithClassTestsButNoCandidateClass.py", new TestCasesShouldContainTestsCheck());
  }
}
