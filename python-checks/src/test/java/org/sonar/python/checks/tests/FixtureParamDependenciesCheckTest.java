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

class FixtureParamDependenciesCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/fixtureParamDependencies_test.py", new FixtureParamDependenciesCheck());
  }

  @Test
  void test_conftest() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/conftest.py", new FixtureParamDependenciesCheck());
  }

  @Test
  void test_non_pytest_file() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/tests/fixtureParamDependenciesNonPytestFile.py", new FixtureParamDependenciesCheck());
  }

  @Test
  void test_pytest_hooks_and_imported_fixture() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/fixtureParamDependenciesPytestHook_test.py", new FixtureParamDependenciesCheck());
  }

  @Test
  void test_scope() {
    assertThat(new FixtureParamDependenciesCheck().scope()).isEqualTo(PythonCheck.CheckScope.ALL);
  }
}
