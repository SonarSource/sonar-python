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
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.types.TypeShed;

class ArgumentTypeCheckTest {

  @AfterEach
  void reset_python_version() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentType.py", new ArgumentTypeCheck());
  }

  @Test
  void test_overloaded_functions() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentType_overloaded_functions.py", new ArgumentTypeCheck());
  }

  @Test
  void importedTypesTest() {
    PythonCheckVerifier.verify(
      List.of(
        "src/test/resources/checks/argumentTypeImporting.py",
        "src/test/resources/checks/argumentTypeImported.py"
      ),
      new ArgumentTypeCheck()
    );
  }

  @Test
  void mapping_types_when_python_version_is_unspecified() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();

    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/argumentType_mapping.py", new ArgumentTypeCheck());
  }

  @Test
  void mapping_types_when_python_version_is_39() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_39));
    TypeShed.resetBuiltinSymbols();

    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/argumentType_mapping.py", new ArgumentTypeCheck());
  }
}
