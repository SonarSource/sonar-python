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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class ModuleNameCheckTest {

  private ModuleNameCheck check = new ModuleNameCheck();

  @Test
  void bad_name() {
    PythonCheckVerifier.verify("src/test/resources/checks/badModule_name.py", check);
  }

  @Test
  void good_name_camel_case() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/ModuleName.py", check);
  }

  @Test
  void good_name_snake_case() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/module_name.py", check);
  }

}
