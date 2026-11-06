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

import java.util.EnumSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.types.TypeShed;

import static org.assertj.core.api.Assertions.assertThat;

class FunctionLevelImportCheckTest {

  private static final String FILE = "src/test/resources/checks/functionLevelImport.py";

  @AfterEach
  void resetPythonVersion() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void python315() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_315));
    PythonCheckVerifier.verify(FILE, new FunctionLevelImportCheck());
  }

  @Test
  void noIssueWhenAnEarlierVersionIsTargeted() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_314, PythonVersionUtils.Version.V_315));
    assertThat(PythonCheckVerifier.issues(FILE, new FunctionLevelImportCheck())).isEmpty();
  }

  @Test
  void noIssueWithDefaultVersions() {
    assertThat(PythonCheckVerifier.issues(FILE, new FunctionLevelImportCheck())).isEmpty();
  }
}
