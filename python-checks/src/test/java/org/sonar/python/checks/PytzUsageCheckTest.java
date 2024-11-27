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

class PytzUsageCheckTest {
  @Test
  void test_39_310_311_312() {
    ProjectPythonVersion
      .setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_39, PythonVersionUtils.Version.V_310, PythonVersionUtils.Version.V_311, PythonVersionUtils.Version.V_312));
    PythonCheckVerifier.verify("src/test/resources/checks/pytzUsage.py", new PytzUsageCheck());
  }
}
