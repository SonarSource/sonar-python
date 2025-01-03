/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks.cdk;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class PublicApiIsSecuritySensitiveCheckTest {

  final PythonCheck check = new PublicApiIsSecuritySensitiveCheck();

  @Test
  void testCfnMethod() {
    PythonCheckVerifier.verify("src/test/resources/checks/cdk/publicApiIsSecuritySensitiveCfnMethodCheck.py", check);
  }

  @Test
  void testCfnRoute() {
    PythonCheckVerifier.verify("src/test/resources/checks/cdk/publicApiIsSecuritySensitiveCfnRouteCheck.py", check);
  }

  @Test
  void testRestApi() {
    PythonCheckVerifier.verify("src/test/resources/checks/cdk/publicApiIsSecuritySensitiveRestApiCheck.py", check);
  }
}
