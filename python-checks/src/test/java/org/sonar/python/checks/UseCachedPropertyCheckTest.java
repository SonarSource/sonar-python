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
import org.sonar.plugins.python.api.PythonVersionUtils.Version;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.types.TypeShed;

import static org.assertj.core.api.Assertions.assertThat;

class UseCachedPropertyCheckTest {

  private static final String FILE = "src/test/resources/checks/useCachedProperty.py";

  @AfterEach
  void resetPythonVersion() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void python312() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(Version.V_312));
    PythonCheckVerifier.verify(FILE, new UseCachedPropertyCheck());
  }

  @Test
  void python311() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(Version.V_311));
    assertThat(PythonCheckVerifier.issues(FILE, new UseCachedPropertyCheck())).isEmpty();
  }

  @Test
  void python311And312() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(Version.V_311, Version.V_312));
    assertThat(PythonCheckVerifier.issues(FILE, new UseCachedPropertyCheck())).isEmpty();
  }

}
