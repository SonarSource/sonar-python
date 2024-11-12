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

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class ArgumentTypeCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentType.py", new ArgumentTypeCheck());
  }

  @Test
  void test_overloaded_functions() {
    PythonCheckVerifier.verify("src/test/resources/checks/argumentType_overloaded_functions.py", new ArgumentTypeCheck());
  }

  @Test
  void importedTypesTest() {
    PythonCheckVerifier.verify(
      List.of(
        "src/test/resources/checks/argumentTypeImporting.py",
        "src/test/resources/checks/argumentTypeImported.py"
      ),
      new ArgumentTypeCheck()
    );
  }
}
