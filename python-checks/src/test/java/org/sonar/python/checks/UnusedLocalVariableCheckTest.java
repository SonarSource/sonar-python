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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UnusedLocalVariableCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/unusedLocalVariable.py", new UnusedLocalVariableCheck());
  }

  @Test
  void custom() {
    UnusedLocalVariableCheck check = new UnusedLocalVariableCheck();
    check.format = "(_|myignore)";
    PythonCheckVerifier.verify("src/test/resources/checks/unusedLocalVariableCustom.py", check);
  }

  @Test
  void tupleQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def using_tuples():\n" +
      "    x, y = (1, 2)\n" +
      "    print x";
    var after = "def using_tuples():\n" +
      "    x, _ = (1, 2)\n" +
      "    print x";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");

    before = "def using_tuples():\n" +
      "    x, y = (1, 2)\n" +
      "    y = 5\n" +
      "    print x";
    after = "def using_tuples():\n" +
      "    x, _ = (1, 2)\n" +
      "    y = 5\n" +
      "    print x";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void exceptClauseQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def foo():\n" +
      "  try:\n" +
      "    ...\n" +
      "  except Type as e:\n" +
      "    ...";
    var after = "def foo():\n" +
      "  try:\n" +
      "    ...\n" +
      "  except Type:\n" +
      "    ...";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Remove the unused local variable");
  }
}
