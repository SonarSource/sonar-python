/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class WildcardImportCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/wildcardImport.py", new WildcardImportCheck());
  }

  @Test
  void test_assignment() {
    PythonCheckVerifier.verify("src/test/resources/checks/wildcardImport/wildcardImport_assignment.py", new WildcardImportCheck());
  }

  @Test
  void test_invalid_assignment() {
    PythonCheckVerifier.verify("src/test/resources/checks/wildcardImport/wildcardImport_invalid_assignment.py", new WildcardImportCheck());
  }

  @Test
  void test_call() {
    PythonCheckVerifier.verify("src/test/resources/checks/wildcardImport/wildcardImport_call.py", new WildcardImportCheck());
  }

  @Test
  void test_invalid_call() {
    PythonCheckVerifier.verify("src/test/resources/checks/wildcardImport/wildcardImport_invalid_call.py", new WildcardImportCheck());
  }

  @Test
  void test_init_py() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/wildcardImport/__init__.py", new WildcardImportCheck());
  }

  @Test
  void test_allowed() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/wildcardImport/wildcardImport_allowed.py", new WildcardImportCheck());
  }

  @Test
  void test_empty() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/wildcardImport/wildcardImport_empty.py", new WildcardImportCheck());
  }

}
