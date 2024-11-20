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

class NotImplementedErrorInOperatorMethodsCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/notImplementedErrorInOperatorMethods.py", new NotImplementedErrorInOperatorMethodsCheck());
  }

  @Test
  void quickFixTest() {
    var check = new NotImplementedErrorInOperatorMethodsCheck();
    String codeWithIssue = "class MyClass:\n" +
      "    def __lt__(self, other):\n" +
      "        raise NotImplementedError()";
    String codeFixed = "class MyClass:\n" +
      "    def __lt__(self, other):\n" +
      "        return NotImplemented";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, NotImplementedErrorInOperatorMethodsCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void quickFixOfExceptionAsVariableTest() {
    var check = new NotImplementedErrorInOperatorMethodsCheck();
    String codeWithIssue = "class MyClass:\n" +
      "    def __lt__(self, other):\n" +
      "        ex = NotImplementedError()\n" +
      "        raise ex";
    String codeFixed = "class MyClass:\n" +
      "    def __lt__(self, other):\n" +
      "        ex = NotImplementedError()\n" +
      "        return NotImplemented";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, NotImplementedErrorInOperatorMethodsCheck.QUICK_FIX_MESSAGE);
  }
}
