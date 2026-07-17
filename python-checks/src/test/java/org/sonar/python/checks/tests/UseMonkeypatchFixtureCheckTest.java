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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.junit.jupiter.api.Assertions.assertEquals;

class UseMonkeypatchFixtureCheckTest {

  private static final String RESOURCES = "src/test/resources/checks/tests/";

  @Test
  void test() {
    PythonCheckVerifier.verify(RESOURCES + "test_useMonkeypatchFixture.py", new UseMonkeypatchFixtureCheck());
  }

  @ParameterizedTest
  @CsvSource({
    "useMonkeypatchFixture.py, false",
    "useMonkeypatchFixture_test.py, true",
    "test_useMonkeypatchFixture_lifecycle.py, false"
  })
  void testFileNames(String fileName, boolean expectIssues) {
    var check = new UseMonkeypatchFixtureCheck();
    var path = RESOURCES + fileName;
    if (expectIssues) {
      PythonCheckVerifier.verify(path, check);
    } else {
      PythonCheckVerifier.verifyNoIssue(path, check);
    }
  }

  @Test
  void scope() {
    assertEquals(PythonCheck.CheckScope.ALL, new UseMonkeypatchFixtureCheck().scope());
  }
}
