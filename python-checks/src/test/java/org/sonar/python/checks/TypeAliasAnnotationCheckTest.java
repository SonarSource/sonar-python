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

class TypeAliasAnnotationCheckTest {

  @AfterEach
  void reset_python_version() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void verify_python_312_issues() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    PythonCheckVerifier.verify("src/test/resources/checks/typeAliasAnnotation.py", new TypeAliasAnnotationCheck());
  }

  @Test
  void verify_earlier_version_no_issues() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_311, PythonVersionUtils.Version.V_312));
    var issues = PythonCheckVerifier.issues("src/test/resources/checks/typeAliasAnnotation.py", new TypeAliasAnnotationCheck());
    assertThat(issues)
      .isEmpty();
  }

}
