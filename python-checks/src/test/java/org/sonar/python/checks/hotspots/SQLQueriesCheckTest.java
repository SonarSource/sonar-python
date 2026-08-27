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
package org.sonar.python.checks.hotspots;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class SQLQueriesCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/hotspots/sqlQueryDjango.py", new SQLQueriesCheck());
    PythonCheckVerifier.verify("src/test/resources/checks/hotspots/sqlQueryOracleDB.py", new SQLQueriesCheck());
    PythonCheckVerifier.verify("src/test/resources/checks/hotspots/sqlQueryOracleDBFromImport.py", new SQLQueriesCheck());
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/hotspots/sqlQueryNoDjango.py", new SQLQueriesCheck());
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/hotspots/sqlQueryNoOracleDB.py", new SQLQueriesCheck());
  }

  /**
   * The check instance is reused across files during a real analysis (see PythonScanner).
   * This verifies that isUsingOracleDB (and the other file-scoped flags) are properly reset
   * between files instead of leaking from one file to the next.
   */
  @Test
  void test_state_is_reset_between_files() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/hotspots/sqlQueryOracleDB.py",
      "src/test/resources/checks/hotspots/sqlQueryNoOracleDB.py"), new SQLQueriesCheck());
  }
}
