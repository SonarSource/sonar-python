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

class DeadStoreCheckTest {

  private final PythonCheck check = new DeadStoreCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/deadStore.py", check);
  }

  @Test
  void iPythonTest() {
    PythonCheckVerifier.verify("src/test/resources/checks/deadStore.ipynb", check);
  }


  @Test
  void pandasTest() {
    PythonCheckVerifier.verify("src/test/resources/checks/deadStorePandas.py", check);
  }

  @Test
  void assignment_on_single_line() {
    String codeWithIssue = code(
      "def foo():",
      "    x = 42",
      "    x = 0",
      "    print(x)");
    String codeFixed = code(
      "def foo():",
      "    x = 0",
      "    print(x)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, DeadStoreCheck.QUICK_FIX_MESSAGE);
  }

  @Test
  void assignment_with_semicolon_on_single_line() {
    String codeWithIssue = code(
      "def foo():",
      "    x = 42 ;",
      "    x = 0",
      "    print(x)");
    String codeFixed = code(
      "def foo():",
      "    x = 0",
      "    print(x)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void assignment_between_two_siblings() {
    String codeWithIssue = code(
      "def foo():",
      "    y = 0; x = 42; x = 0",
      "    print(x)");
    String codeFixed = code(
      "def foo():",
      "    y = 0; x = 0",
      "    print(x)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void assignment_with_previous_element_on_line() {
    String codeWithIssue = code(
      "def foo():",
      "    y = 0; x = 42",
      "    x = 0",
      "    print(x)");
    String codeFixed = code(
      "def foo():",
      "    y = 0;",
      "    x = 0",
      "    print(x)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void assignment_with_trailing_whitespace() {
    String codeWithIssue = code(
      "def foo():",
      "    x = 42 ",
      "    x = 0",
      "    print(x)");
    String codeFixed = code(
      "def foo():",
      "    x = 0",
      "    print(x)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void assignment_wrapped_by_elements_on_different_lines() {
    String codeWithIssue = code(
      "def foo():",
      "    a = 1",
      "    x = 10",
      "    x = 0",
      "    print(x)");
    String codeFixed = code(
      "def foo():",
      "    a = 1",
      "    x = 0",
      "    print(x)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void all_statements_on_single_line() {
    String codeWithIssue = "def dead_store(): unused = 24; unused = 42; print(unused)";
    String codeFixed = "def dead_store(): unused = 42; print(unused)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void assignment_before_condition() {
    String codeWithIssue = code(
      "def simple_conditional():",
      "    x = 10",
      "    if p:",
      "        x = 11",
      "        print(x)");
    String codeFixed = code(
      "def simple_conditional():",
      "    if p:",
      "        x = 11",
      "        print(x)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void function_call_as_assigned_value() {
    String codeWithIssue = code(
      "def function_assign():",
      "    c = foo()",
      "    print(c)",
      "    c = foo()",
      "");
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void assignment_expression_in_function_call() {
    String codeWithIssue = code(
      "def assignment_expression():",
      "    foo(a:=3) # Comment 1",
      "# Comment 2",
      "    a = 2",
      "    print(a)");
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }


  @Test
  void comment_after_should_not_be_removed(){
    String codeWithIssue = code(
      "def ab():",
      "    a = 42",
      "    # This is an important comment",
      "    a = 43",
      "    print(a)");
    String codeFixed = code(
      "def ab():",
      "    # This is an important comment",
      "    a = 43",
      "    print(a)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void new_line_after_assignment_should_not_be_removed(){
    String codeWithIssue = code(
      "def ab():",
      "    b = 1",
      "    a = 42",
      "",
      "    # This is an important comment",
      "    a = 43",
      "    print(a)");
    String codeFixed = code(
      "def ab():",
      "    b = 1",
      "",
      "    # This is an important comment",
      "    a = 43",
      "    print(a)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void assignment_on_branch(){
    String codeWithIssue = code(
      "def a():",
      "    x = 42",
      "    if x:",
      "        x = 43",
      "    print(a)");
    String codeFixed = code(
      "def a():",
      "    x = 42",
      "    if x:",
      "        pass",
      "    print(a)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void assignment_on_branch_with_multiple_elements(){
    String codeWithIssue = code(
      "def a():",
      "    x = 42",
      "    if x:",
      "        x = 43",
      "        a = 4",
      "    print(a)");
    String codeFixed = code(
      "def a():",
      "    x = 42",
      "    if x:",
      "        a = 4",
      "    print(a)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void assignment_on_branch_with_multiple_elements_on_same_line(){
    String codeWithIssue = code(
      "def a():",
      "    x = 42",
      "    if x:",
      "        a = 4; x = 43",
      "    print(a)");
    String codeFixed = code(
      "def a():",
      "    x = 42",
      "    if x:",
      "        a = 4;",
      "    print(a)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void side_effect_in_binary_op(){
    String codeWithIssue = code(
      "def ab():",
      "    a = 1 + foo()",
      "    a = 2",
      "    print(a)");
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }

  @Test
  void chain_assignment() {
    var codeWithIssue = code(
      "def chain_assign():",
        "    a = b = 42",
        "    b = foo()",
        "    print(a, b)"
    );
    var codeFixed = code(
      "def chain_assign():",
      "    a = 42",
      "    b = foo()",
      "    print(a, b)"
    );
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = code(
      "def chain_assign():",
      "    a = b = 42",
      "    a = foo()",
      "    print(a, b)"
    );
    codeFixed = code(
      "def chain_assign():",
      "    b = 42",
      "    a = foo()",
      "    print(a, b)"
    );
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }
}
