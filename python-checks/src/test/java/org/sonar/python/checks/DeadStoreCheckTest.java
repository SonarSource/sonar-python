/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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

import org.junit.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.sonar.python.checks.utils.CodeTestUtils.code;

public class DeadStoreCheckTest {

  private final PythonCheck check = new DeadStoreCheck();

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/deadStore.py", check);
  }

  @Test
  public void quickfix() {
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
  }

  @Test
  public void semicolon() {
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
  public void space() {
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
  public void quickfix_after_non_issue() {
    String codeWithIssue = code(
      "def foo():",
      "    a = 1",
      "    x = 10 ;",
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
  public void quickfix_oneline() {
    String codeWithIssue = "def dead_store(): unused = 24; unused = 42; print(unused)";
    String codeFixed = "def dead_store(): unused = 42; print(unused)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void quickfix_in_condition() {
    String codeWithIssue = code(
      "def simple_conditional():",
      "    x = 10 # Noncompliant",
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
  public void unused_after_reassignment() {
    String codeWithIssue = code(
      "def tuple_assign():",
      "    c = foo()",
      "    print(c)",
      "    c = foo()",
      "");
    String codeFixed = code(
      "def tuple_assign():",
      "    c = foo()",
      "    print(c)",
      "");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void end_of_line_comments_should_be_removed() {
    String codeWithIssue = code(
      "def assignment_expression():",
      "    foo(a:=3) # Comment 1",
      "# Comment 2",
      "    a = 2",
      "    print(a)");
    String codeFixed = code(
      "def assignment_expression():",
      "    # Comment 2",
      "    a = 2",
      "    print(a)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void no_separator_found(){
    String codeWithIssue = code(
      "def ab():",
      "    a = foo()",
      "    print(a)",
      "    a = foo()",
      "");
    String codeFixed = code(
      "def ab():",
      "    a = foo()",
      "    print(a)",
      "");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void one_separator_found(){
    String codeWithIssue = code(
      "def ab():",
      "    a = foo()",
      "    print(a)",
      "    a = foo();");
    String codeFixed = code(
      "def ab():",
      "    a = foo()",
      "    print(a)\n");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void two_separators_found(){
    String codeWithIssue = code(
      "def ab():",
      "    a = foo()",
      "    print(a)",
      "    a = foo();\n");
    String codeFixed = code(
      "def ab():",
      "    a = foo()",
      "    print(a)\n");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void comment_after_should_not_be_removed(){
    String codeWithIssue = code(
      "def ab():",
      "    a = 42",
      "    # This is an important comment",
      "    a = 43",
      "    print(a)");
    String codeFixed = code(
      "def ab():",
      "        # This is an important comment",
      "    a = 43",
      "    print(a)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void space_comment_after_should_not_be_removed(){
    String codeWithIssue = code(
      "def ab():",
      "    b = 1",
      "    a = 42",
      "\n"+
      "    # This is an important comment",
      "    a = 43",
      "    print(a)");
    String codeFixed = code(
      "def ab():",
      "    b = 1",
      "    \n"+
      "    # This is an important comment",
      "    a = 43",
      "    print(a)");
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void deadstore_one_branch(){
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
}
