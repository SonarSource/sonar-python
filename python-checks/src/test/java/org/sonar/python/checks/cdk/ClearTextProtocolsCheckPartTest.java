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

class ClearTextProtocolsCheckPartTest {

  final PythonCheck check = new ClearTextProtocolsCheckPart();

  @Test
  void elb() {
    PythonCheckVerifier.verify("src/test/resources/checks/cdk/clearTextProtocolsCheck_elb.py", check);
  }
  @Test
  void elbv2() {
    PythonCheckVerifier.verify("src/test/resources/checks/cdk/clearTextProtocolsCheck_elbv2.py", check);
  }

  @Test
  void elasticache() {
    PythonCheckVerifier.verify("src/test/resources/checks/cdk/clearTextProtocolsCheck_elasticache.py", check);
  }

  @Test
  void kinesis() {
    PythonCheckVerifier.verify("src/test/resources/checks/cdk/clearTextProtocolsCheck_kinesis.py", check);
  }
}
