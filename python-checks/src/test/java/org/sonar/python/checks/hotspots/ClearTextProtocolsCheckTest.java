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

class ClearTextProtocolsCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/hotspots/clearTextProtocols.py",
      "src/test/resources/checks/hotspots/clearTextProtocols_httpServerTls.py",
      "src/test/resources/checks/hotspots/clearTextProtocols_httpServerModuleScope.py"), new ClearTextProtocolsCheck());
  }

}
