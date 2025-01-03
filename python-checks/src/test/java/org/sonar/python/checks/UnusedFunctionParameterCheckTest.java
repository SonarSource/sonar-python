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
package org.sonar.python.checks;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UnusedFunctionParameterCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/unusedFunctionParameter/unusedFunctionParameter.py", new UnusedFunctionParameterCheck());
  }

  @Test
  void test_import() {
    PythonCheckVerifier.verify(
      List.of(
        "src/test/resources/checks/unusedFunctionParameter/unusedFunctionParameterImport.py",
        "src/test/resources/checks/unusedFunctionParameter/unusedFunctionParameterImported.py"
      ),
      new UnusedFunctionParameterCheck()
    );
  }

  @Test
  void test_conftest() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/unusedFunctionParameter/conftest.py", new UnusedFunctionParameterCheck());
  }

  @Test
  void test_test_file() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/unusedFunctionParameter/test_something.py", new UnusedFunctionParameterCheck());
  }

}
