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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.sonar.python.checks.utils.CodeTestUtils.code;

class ClassMethodFirstArgumentNameCheckTest {

  @Test
  void testRule() {
    PythonCheckVerifier.verify("src/test/resources/checks/classMethodFirstArgumentNameCheck.py", new ClassMethodFirstArgumentNameCheck());
  }

  @Test
  void testQuickFix() {
    String codeWithIssue = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(bob, alice):",
      "        print(bob)");
    String codeFixed1 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(cls, bob, alice):",
      "        print(bob)");
    String codeFixed2 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(cls, alice):",
      "        print(cls)");

    PythonQuickFixVerifier.verify(new ClassMethodFirstArgumentNameCheck(), codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void testQuickFixMultiline() {
    String codeWithIssue = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(bob,",
      "            alice):",
      "        print(bob)");
    String codeFixed1 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(cls,",
      "            bob,",
      "            alice):",
      "        print(bob)");
    String codeFixed2 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(cls,",
      "            alice):",
      "        print(cls)");

    PythonQuickFixVerifier.verify(new ClassMethodFirstArgumentNameCheck(), codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void testQuickFixMultiline2() {
    String codeWithIssue = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            bob,",
      "            alice):",
      "        print(bob)");
    String codeFixed1 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            cls,",
      "            bob,",
      "            alice):",
      "        print(bob)");
    String codeFixed2 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            cls,",
      "            alice):",
      "        print(cls)");

    PythonQuickFixVerifier.verify(new ClassMethodFirstArgumentNameCheck(), codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void testQuickFixMultiline3() {
    String codeWithIssue = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            bob, alice",
      "    ):",
      "        print(bob)");
    String codeFixed1 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            cls, bob, alice",
      "    ):",
      "        print(bob)");
    String codeFixed2 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            cls, alice",
      "    ):",
      "        print(cls)");

    PythonQuickFixVerifier.verify(new ClassMethodFirstArgumentNameCheck(), codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void testQuickFixMultiline4() {
    String codeWithIssue = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            bob",
      "    ):",
      "        print(bob)");
    String codeFixed1 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            cls, bob",
      "    ):",
      "        print(bob)");
    String codeFixed2 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            cls",
      "    ):",
      "        print(cls)");

    PythonQuickFixVerifier.verify(new ClassMethodFirstArgumentNameCheck(), codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void testQuickFixMultilineCustomClassParameterNames() {
    ClassMethodFirstArgumentNameCheck check = new ClassMethodFirstArgumentNameCheck();
    check.classParameterNames = "xxx";
    String codeWithIssue = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            bob, alice",
      "    ):",
      "        print(bob)");
    String codeFixed1 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            xxx, bob, alice",
      "    ):",
      "        print(bob)");
    String codeFixed2 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            xxx, alice",
      "    ):",
      "        print(xxx)");

    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void testQuickFixMultilineEmptyClassParameterNames() {
    ClassMethodFirstArgumentNameCheck check = new ClassMethodFirstArgumentNameCheck();
    check.classParameterNames = "";
    String codeWithIssue = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            bob, alice",
      "    ):",
      "        print(bob)");
    String codeFixed1 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            cls, bob, alice",
      "    ):",
      "        print(bob)");
    String codeFixed2 = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            cls, alice",
      "    ):",
      "        print(cls)");

    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void testQuickFixDescriptions() {
    ClassMethodFirstArgumentNameCheck check = new ClassMethodFirstArgumentNameCheck();
    check.classParameterNames = "xxx";
    String codeWithIssue = code(
      "class A():",
      "    @classmethod",
      "    def long_function_name(",
      "            bob, alice",
      "    ):",
      "        print(bob)");

    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue,
      "Add 'xxx' as the first argument.",
      "Rename 'bob' to 'xxx'");
  }
}
