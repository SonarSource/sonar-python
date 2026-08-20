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
package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class HardcodedIPCheckTest {

  @Test
  void reports_hardcoded_ips_in_non_test_files() {
    PythonCheckVerifier.verify("src/test/resources/checks/hardcodedIP.py", new HardcodedIPCheck());
  }

  /** Verifies likely test files do not report hardcoded IP addresses. */
  @Test
  void skips_likely_test_files() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/hardcodedIPDjango.py", new HardcodedIPCheck());
  }
}
