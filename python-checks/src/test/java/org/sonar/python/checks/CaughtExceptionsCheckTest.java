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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class CaughtExceptionsCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/caughtExceptions.py", new CaughtExceptionsCheck());
  }

  @Test
  void quickFixTest() {
    var check = new CaughtExceptionsCheck();
    var before = "class CustomException:\n" +
      "  ...\n" +
      "\n" +
      "def foo():\n" +
      "  try:\n" +
      "    a = bar()\n" +
      "  except CustomException:\n" +
      "    print(\"Exception\")";

    var after = "class CustomException(Exception):\n" +
      "  ...\n" +
      "\n" +
      "def foo():\n" +
      "  try:\n" +
      "    a = bar()\n" +
      "  except CustomException:\n" +
      "    print(\"Exception\")";

    var expectedMessage = String.format(CaughtExceptionsCheck.QUICK_FIX_MESSAGE_FORMAT, "CustomException");

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @Test
  void quickFixIPythonTest() {
    var check = new CaughtExceptionsCheck();
    var before = "#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n" +
      "class CustomException:\n" +
      "  ...\n" +
      "\n" +
      "#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n" +
      "def foo():\n" +
      "  try:\n" +
      "    a = %time bar()\n" +
      "  except CustomException:\n" +
      "    ...";

    var after = "#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n" +
      "class CustomException(Exception):\n" +
      "  ...\n" +
      "\n" +
      "#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER\n" +
      "def foo():\n" +
      "  try:\n" +
      "    a = %time bar()\n" +
      "  except CustomException:\n" +
      "    ...";

    var expectedMessage = String.format(CaughtExceptionsCheck.QUICK_FIX_MESSAGE_FORMAT, "CustomException");

    PythonQuickFixVerifier.verifyIPython(check, before, after);
    PythonQuickFixVerifier.verifyIPythonQuickFixMessages(check, before, expectedMessage);
  }

  @Test
  void exceptionWithEmptyParenthesisQuickFixTest() {
    var check = new CaughtExceptionsCheck();
    var before = "class CustomException():\n" +
      "  ...\n" +
      "\n" +
      "def foo():\n" +
      "  try:\n" +
      "    a = bar()\n" +
      "  except CustomException:\n" +
      "    print(\"Exception\")";

    var after = "class CustomException(Exception):\n" +
      "  ...\n" +
      "\n" +
      "def foo():\n" +
      "  try:\n" +
      "    a = bar()\n" +
      "  except CustomException:\n" +
      "    print(\"Exception\")";

    var expectedMessage = String.format(CaughtExceptionsCheck.QUICK_FIX_MESSAGE_FORMAT, "CustomException");

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @Test
  void exceptionWithNotEmptyParenthesisQuickFixTest() {
    var check = new CaughtExceptionsCheck();
    var before = "class AbcEx:\n" +
      "    ...\n" +
      "\n" +
      "class Ex(AbcEx):\n" +
      "    ...\n" +
      "\n" +
      "def someLogic():\n" +
      "    try:\n" +
      "        a = foo()\n" +
      "    except Ex:\n" +
      "        ...";

    var after = "class AbcEx:\n" +
      "    ...\n" +
      "\n" +
      "class Ex(AbcEx, Exception):\n" +
      "    ...\n" +
      "\n" +
      "def someLogic():\n" +
      "    try:\n" +
      "        a = foo()\n" +
      "    except Ex:\n" +
      "        ...";

    var expectedMessage = String.format(CaughtExceptionsCheck.QUICK_FIX_MESSAGE_FORMAT, "Ex");

    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }
}
