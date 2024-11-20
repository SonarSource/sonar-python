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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UndefinedNameAllPropertyCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/undefinedNameAllProperty/undefinedNameAllProperty.py", new UndefinedNameAllPropertyCheck());
  }

  @Test
  void test_wildcard_import() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/undefinedNameAllProperty/with_wildcard_import.py", new UndefinedNameAllPropertyCheck());
  }

  @Test
  void test_init_exports_module() {
    PythonCheckVerifier.verify("src/test/resources/checks/undefinedNameAllProperty/__init__.py", new UndefinedNameAllPropertyCheck());
  }

  @Test
  void test_uses_getattr() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/undefinedNameAllProperty/defines_getattr.py", new UndefinedNameAllPropertyCheck());
  }

  @Test
  void test_manipulates_globals() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/undefinedNameAllProperty/manipulates_globals.py", new UndefinedNameAllPropertyCheck());
  }

  @Test
  void test_manipulates_sys_modules() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/undefinedNameAllProperty/manipulates_sys_modules.py", new UndefinedNameAllPropertyCheck());
  }

}
