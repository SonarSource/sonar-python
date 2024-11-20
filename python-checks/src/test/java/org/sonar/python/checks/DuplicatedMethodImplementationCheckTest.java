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

import static org.sonar.python.checks.utils.CodeTestUtils.code;

class DuplicatedMethodImplementationCheckTest {

  public static final DuplicatedMethodImplementationCheck CHECK = new DuplicatedMethodImplementationCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/duplicatedMethodImplementationCheck.py", CHECK);
  }

  @Test
  void testQuickFixSimple() {
    String code = code(
      "class clazz:",
      "  def method(self):",
      "    foo()",
      "    bar()",
      "",
      "  def method2(self):",
      "    foo()",
      "    bar()",
      "");
    String fixedCode = code(
      "class clazz:",
      "  def method(self):",
      "    foo()",
      "    bar()",
      "",
      "  def method2(self):",
      "    self.method()",
      "");

    PythonQuickFixVerifier.verify(CHECK, code, fixedCode);
  }

  @Test
  void testQuickFixWithReturnedValue() {
    String code = code(
      "class clazz:",
      "  def method(self):",
      "    print(1)",
      "    return 42",
      "",
      "  def method2(self):",
      "    print(1)",
      "    return 42",
      "");
    String fixedCode = code(
      "class clazz:",
      "  def method(self):",
      "    print(1)",
      "    return 42",
      "",
      "  def method2(self):",
      "    return self.method()",
      "");

    PythonQuickFixVerifier.verify(CHECK, code, fixedCode);
  }

  @Test
  void testQuickFixWithIfExpression() {
    String code = code(
      "class clazz:",
      "  def method_1(self):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "",
      "  def method_2(self):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "");
    String fixedCode = code(
      "class clazz:",
      "  def method_1(self):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "",
      "  def method_2(self):",
      "    self.method_1()");

    PythonQuickFixVerifier.verify(CHECK, code, fixedCode);
  }
  @Test
  void testQuickFixWithWithoutArguments() {
    String code = code(
      "class clazz:",
      "  def method_1():",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "",
      "  def method_2():",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "");
    String fixedCode = code(
      "class clazz:",
      "  def method_1():",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "",
      "  def method_2():",
      "    self.method_1()");

    PythonQuickFixVerifier.verify(CHECK, code, fixedCode);
  }
  @Test
  void testQuickFixWithCustomDecorator() {
    String code = code(
      "class clazz:",
      "  @some_decorator",
      "  def method_1(self):",
      "    print(10)",
      "    print(20)",
      "",
      "  @some_decorator",
      "  def method_2(self):",
      "    print(10)",
      "    print(20)",
      "");
    String fixedCode = code(
      "class clazz:",
      "  @some_decorator",
      "  def method_1(self):",
      "    print(10)",
      "    print(20)",
      "",
      "  @some_decorator",
      "  def method_2(self):",
      "    self.method_1()",
      "");

    PythonQuickFixVerifier.verify(CHECK, code, fixedCode);
  }

  @Test
  void testNoQuickFixWhenSomeArguments() {
    String code = code(
      "class clazz:",
      "  def method_1(self, text):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "",
      "  def method_2(self, text):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "");

    PythonQuickFixVerifier.verifyNoQuickFixes(CHECK, code);
  }
  @Test
  void testNoQuickFixWhenSomeArgumentsButNotSelf() {
    String code = code(
      "class clazz:",
      "  def method_1(text):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "",
      "  def method_2(text):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "");

    PythonQuickFixVerifier.verifyNoQuickFixes(CHECK, code);
  }
  @Test
  void testNoQuickFixWhenClassMethodAndClsAsFirstArg() {
    String code = code(
      "class clazz:",
      "  @classmethod",
      "  def method_1(cls):",
      "    print(10)",
      "    print(20)",
      "",
      "  @classmethod",
      "  def method_2(cls):",
      "    print(10)",
      "    print(20)",
      "");

    String fixedCode = code(
      "class clazz:",
      "  @classmethod",
      "  def method_1(cls):",
      "    print(10)",
      "    print(20)",
      "",
      "  @classmethod",
      "  def method_2(cls):",
      "    clazz.method_1()",
      "");

    PythonQuickFixVerifier.verify(CHECK, code, fixedCode);
  }
  @Test
  void testNoQuickFixForStaticMethod() {
    String code = code(
      "class clazz:",
      "  @staticmethod",
      "  def method_1(cls):",
      "    print(10)",
      "    print(20)",
      "",
      "  @staticmethod",
      "  def method_2(cls):",
      "    print(10)",
      "    print(20)",
      "");

    String fixedCode = code(
      "class clazz:",
      "  @staticmethod",
      "  def method_1(cls):",
      "    print(10)",
      "    print(20)",
      "",
      "  @staticmethod",
      "  def method_2(cls):",
      "    clazz.method_1()",
      "");

    PythonQuickFixVerifier.verify(CHECK, code, fixedCode);
  }

  @Test
  void testNoQuickFixWhenClassMethodWithArguments() {
    String code = code(
      "class clazz:",
      "  @classmethod",
      "  def method_1(cls, text):",
      "    print(10)",
      "    print(20)",
      "",
      "  @classmethod",
      "  def method_2(cls, text):",
      "    print(10)",
      "    print(20)",
      "");

    PythonQuickFixVerifier.verifyNoQuickFixes(CHECK, code);
  }

  @Test
  void testQuickFixMessage() {
    String code = code(
      "class clazz:",
      "  def method_1(self):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "",
      "  def method_2(self):",
      "    if cond:",
      "      foo()",
      "    else:",
      "      bar()",
      "");

    PythonQuickFixVerifier.verifyQuickFixMessages(CHECK, code, "Call method_1 inside this function.");
  }
}
