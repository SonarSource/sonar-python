/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UselessStatementCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/uselessStatement/uselessStatement.py", new UselessStatementCheck());
  }

  @Test
  void custom() {
    UselessStatementCheck check = new UselessStatementCheck();
    check.reportOnStrings = true;
    check.ignoredOperators = "<<,+";
    PythonCheckVerifier.verify("src/test/resources/checks/uselessStatement/uselessStatementCustom.py", check);
  }

  @Test
  void test_import() {
    PythonCheckVerifier.verify(
      List.of("src/test/resources/checks/uselessStatement/uselessStatementImported.py", "src/test/resources/checks/uselessStatement/uselessStatementImport.py"),
      new UselessStatementCheck()
    );
  }

  @Test
  void test_ignore_manifest() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/__manifest__.py", new UselessStatementCheck());
  }
}
