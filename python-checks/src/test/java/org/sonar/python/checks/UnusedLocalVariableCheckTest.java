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
  void pandasTest() {
    PythonCheckVerifier.verify("src/test/resources/checks/unusedLocalVariablePandas.py", new UnusedLocalVariableCheck());
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

  @Test
  void loopIndexQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def loop_index():\n"
      + "  for i in range(10):\n" +
      "    print(\"Hello\")";
    var after = "def loop_index():\n"
      + "  for _ in range(10):\n" +
      "    print(\"Hello\")";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void loopIndexComprehensionQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def loop_index():\n" + " return [True for i in range(10)]\n";
    var after = "def loop_index():\n" + " return [True for _ in range(10)]\n";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void loopQuickFixIndexAlreadyTakenTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def a():\n" +
      "    _ = 3\n" +
      "    for i in range(10):\n" +
      "        ...\n" +
      "    return _\n";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  @Test
  void loopQuickFixIndexFileLevelAlreadyTakenTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "_ = 42\n" +
      "def foo():\n" +
      "  for i in range(5):\n" +
      "    print(\"hello\")\n" +
      "  print(_)\n" +
      "foo()";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

  @Test
  void loopIndexComprehensionClassQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "class A():\n" +
      "  _ = True\n" +
      "  def __init__(self):\n" +
      "    for i in range(5):\n" +
      "      print(\"print\")";
    var after = "class A():\n" +
      "  _ = True\n" +
      "  def __init__(self):\n" +
      "    for _ in range(5):\n" +
      "      print(\"print\")";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void assignmentQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def foo():\n" +
      "  x = bar()\n" +
      "  y = True\n" +
      "  return y";
    var after = "def foo():\n" +
      "  bar()\n" +
      "  y = True\n" +
      "  return y";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Remove assignment target");
  }

  @Test
  void multipleAssignmentQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def foo():\n" +
      "  x, y, z = bar(), True, False\n" +
      "  return y, z";
    var after = "def foo():\n" +
      "  _, y, z = bar(), True, False\n" +
      "  return y, z";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Replace with \"_\"");
  }

  @Test
  void typeAnnotationQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def foo():\n" +
      "  value: str = \"hello\"\n" +
      "  return [int(value) for value in something()]";
    var after = "def foo():\n" +
      "  \"hello\"\n" +
      "  return [int(value) for value in something()]";

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Remove assignment target");
  }

  @Test
  void typeAnnotationSeparateDeclarationAssignmentNoQuickFixTest() {
    var check = new UnusedLocalVariableCheck();

    var before = "def foo():\n" +
      "  value: str \n" +
      "  value = \"Hello\"\n" +
      "  return [int(value) for value in something()]";

    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }
}
