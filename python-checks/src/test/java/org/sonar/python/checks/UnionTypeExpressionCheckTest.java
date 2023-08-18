/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks;

import java.util.EnumSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class UnionTypeExpressionCheckTest {

  @AfterEach
  void reset_python_version() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
  }

  @Test
  void test_python_3_10() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_310));
    PythonCheckVerifier.verify("src/test/resources/checks/unionTypeExpression.py", new UnionTypeExpressionCheck());
  }

  @Test
  void test_python_3_9() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_39, PythonVersionUtils.Version.V_310));
    var issues = PythonCheckVerifier.issues("src/test/resources/checks/unionTypeExpression.py", new UnionTypeExpressionCheck());
    assertThat(issues)
      .isEmpty();
  }

  @Test
  void test_python_unknown_version() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    var issues = PythonCheckVerifier.issues("src/test/resources/checks/unionTypeExpression.py", new UnionTypeExpressionCheck());
    assertThat(issues)
      .isEmpty();
  }
}
