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
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.sonar.python.checks.utils.CodeTestUtils.code;

class ImplicitStringConcatenationCheckTest {

  private static final PythonCheck check = new ImplicitStringConcatenationCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/implicitStringConcatenation.py", check);
  }

  @Test
  void simple_expression_quickfix() {
    String codeWithIssue = "a = '1' '2'";
    String codeFixed1 = "a = '1' + '2'";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1);

    codeWithIssue = "a = ['1' '2']";
    codeFixed1 = "a = ['1', '2']";
    String codeFixed2 = "a = ['1' + '2']";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);

    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue,
      "Add the comma between string or byte tokens.",
      "Make the addition sign between string or byte tokens explicit.");
  }

  @Test
  void function_statement_quickfix() {
    String codeWithIssue = code(
      "def a():",
      "    b = ['1' '2']",
      "");

    String codeFixed1 = code(
      "def a():",
      "    b = ['1', '2']",
      "");
    String codeFixed2 = code(
      "def a():",
      "    b = ['1' + '2']",
      "");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void concat_expression() {
    String codeWithIssue = "['1'+'2' '3'+'4']";
    String codeFixed1 = "['1'+'2', '3'+'4']";
    String codeFixed2 = "['1'+'2' + '3'+'4']";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void sets() {
    String codeWithIssue = "{'1' '2', '3'}";
    String codeFixed1 = "{'1', '2', '3'}";
    String codeFixed2 = "{'1' + '2', '3'}";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void parameters() {
    String codeWithIssue = "print('1' '2', '3')";
    String codeFixed1 = "print('1', '2', '3')";
    String codeFixed2 = "print('1' + '2', '3')";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void parameters2() {
    String codeWithIssue = "foo('1' '2', '3')";
    String codeFixed1 = "foo('1', '2', '3')";
    String codeFixed2 = "foo('1' + '2', '3')";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void multiline_list() {
    String codeWithIssue = code("a = [",
      "  '1'",
      "  '2'",
      "]");
    String codeFixed1 = code("a = [",
      "  '1',",
      "  '2'",
      "]");
    String codeFixed2 = code("a = [",
      "  '1' + '2'",
      "]");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  void multiline_list_with_comment() {
    String codeWithIssue = code("a = [",
      "  'a',",
      "  'b',",
      "  'c',",
      "  'd'  # A comment",
      "  'e', # Another comment",
      "  'f']");
    String codeFixed1 = code("a = [",
      "  'a',",
      "  'b',",
      "  'c',",
      "  'd',  # A comment",
      "  'e', # Another comment",
      "  'f']");
    String codeFixed2 = code("a = [",
      "  'a',",
      "  'b',",
      "  'c',",
      "  'd' + 'e', # Another comment",
      "  'f']");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }
}
