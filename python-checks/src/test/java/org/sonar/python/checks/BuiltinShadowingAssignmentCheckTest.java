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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class BuiltinShadowingAssignmentCheckTest {

  @Test
  void testVariableShadowing() {
    PythonCheckVerifier.verify("src/test/resources/checks/builtinShadowing.py", new BuiltinShadowingAssignmentCheck());
  }

  @Test
  void quickFixTest() {
    var before = "def my_function():\n" +
      "  int = 42\n" +
      "  print(int)\n" +
      "  return int";

    var after = "def my_function():\n" +
      "  _int = 42\n" +
      "  print(_int)\n" +
      "  return _int";
    var check = new BuiltinShadowingAssignmentCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, String.format(BuiltinShadowingAssignmentCheck.QUICK_FIX_MESSAGE_FORMAT, "int"));
  }

  @Test
  void noQuickFixTest() {
    var check = new BuiltinShadowingAssignmentCheck();

    var before = "def my_function():\n" +
      "  _int = 22\n" +
      "  int = 42\n" +
      "  print(int)\n" +
      "  return int";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);

    before = "def my_function(_int = 22):\n" +
      "  int = 42\n" +
      "  print(int)\n" +
      "  return int";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);

    before = "def my_function((_int, b)):\n" +
      "  int = 42\n" +
      "  print(int)\n" +
      "  return int";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

}
