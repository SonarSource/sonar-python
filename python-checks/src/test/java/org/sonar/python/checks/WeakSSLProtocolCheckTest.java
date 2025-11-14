/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

class WeakSSLProtocolCheckTest {

  @AfterEach
  void reset_python_version() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
  }

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/weakSSLProtocol.py", new WeakSSLProtocolCheck());
  }

  @Test
  void test_python_310() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_310));
    PythonCheckVerifier.verify("src/test/resources/checks/weakSSLProtocol_python310.py", new WeakSSLProtocolCheck());
  }

  @Test
  void test_fallback_import() {
    PythonCheckVerifier.verify("src/test/resources/checks/weakSSLProtocol_fallback_import.py", new WeakSSLProtocolCheck());
  }

  @Test
  void test_apigateway() {
    PythonCheckVerifier.verify("src/test/resources/checks/weakSSLProtocol_apigateway.py", new WeakSSLProtocolCheck());
  }

  @Test
  void test_elasticopensearch() {
    PythonCheckVerifier.verify("src/test/resources/checks/weakSSLProtocol_elastic_and_open_search.py", new WeakSSLProtocolCheck());
  }
}
