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

public class DeadStoreCheckTest {

  private final PythonCheck check = new DeadStoreCheck();

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/deadStore.py", check);
  }

  @Test
  public void quickfix() {
    String codeWithIssue = "def foo():\n" +
      "    x = 42\n" +
      "    x = 0\n" +
      "    print(x)";
    String codeFixed = "def foo():\n" +
      "    x = 0\n" +
      "    print(x)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void semicolon() {
    String codeWithIssue = "def foo():\n" +
      "    x = 42 ;\n" +
      "    x = 0\n" +
      "    print(x)";
    String codeFixed = "def foo():\n" +
      "    x = 0\n" +
      "    print(x)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void space() {
    String codeWithIssue = "def foo():\n" +
      "    x = 42 \n" +
      "    x = 0\n" +
      "    print(x)";
    String codeFixed = "def foo():\n" +
      "    x = 0\n" +
      "    print(x)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void quickfix_after_non_issue() {
    String codeWithIssue = "def foo():\n" +
      "    a = 1\n" +
      "    x = 10 ;\n" +
      "    x = 0\n" +
      "    print(x)";
    String codeFixed = "def foo():\n" +
      "    a = 1\n" +
      "    x = 0\n" +
      "    print(x)";
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
    String codeWithIssue = "def simple_conditional():\n" +
      "    x = 10 # Noncompliant\n" +
      "    if p:\n" +
      "        x = 11\n" +
      "        print(x)";
    String codeFixed = "def simple_conditional():\n" +
      "    if p:\n" +
      "        x = 11\n" +
      "        print(x)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void unused_after_reassignment() {
    String codeWithIssue = "def tuple_assign():\n" +
      "    c = foo()\n" +
      "    print(c)\n" +
      "    c = foo()\n";
    String codeFixed = "def tuple_assign():\n" +
      "    c = foo()\n" +
      "    print(c)\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void end_of_line_comments_should_be_removed() {
    String codeWithIssue = "" +
      "def assignment_expression():\n" +
      "    foo(a:=3) # Comment 1\n" +
      "# Comment 2\n" +
      "    a = 2\n" +
      "    print(a)";
    String codeFixed = "" +
      "def assignment_expression():\n" +
      "    # Comment 2\n" +
      "    a = 2\n" +
      "    print(a)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void no_separator_found(){
    String codeWithIssue = "" +
      "def ab():\n" +
      "    a = foo()\n" +
      "    print(a)\n" +
      "    a = foo()";
    String codeFixed = "" +
      "def ab():\n" +
      "    a = foo()\n" +
      "    print(a)\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void one_separator_found(){
    String codeWithIssue = "" +
      "def ab():\n" +
      "    a = foo()\n" +
      "    print(a)\n" +
      "    a = foo();";
    String codeFixed = "" +
      "def ab():\n" +
      "    a = foo()\n" +
      "    print(a)\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  public void two_separators_found(){
    String codeWithIssue = "" +
      "def ab():\n" +
      "    a = foo()\n" +
      "    print(a)\n" +
      "    a = foo();\n";
    String codeFixed = "" +
      "def ab():\n" +
      "    a = foo()\n" +
      "    print(a)\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }
}
