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

import java.util.EnumSet;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class GenericTypeStatementCheckTest {

  private static final GenericTypeStatementCheck check = new GenericTypeStatementCheck();

  @Test
  void test_1() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_312));
    PythonCheckVerifier.verify("src/test/resources/checks/genericTypeStatement.py", check);
  }

  @Test
  void test_2() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_311, PythonVersionUtils.Version.V_312));
    var issues = PythonCheckVerifier.issues("src/test/resources/checks/genericTypeStatement.py", check);
    assertThat(issues)
      .isEmpty();
  }
}
