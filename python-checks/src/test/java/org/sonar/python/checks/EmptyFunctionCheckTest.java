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
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.sonar.python.checks.utils.CodeTestUtils.code;

public class EmptyFunctionCheckTest {

  final PythonCheck check = new EmptyFunctionCheck();

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/emptyFunction.py", check);
  }

  @Test
  public void quick_fixes_for_function() {
    String codeWithIssue = code("def my_function():",
      "  pass");
    String addComment = code("def my_function():",
      "  # TODO document why this method is empty",
      "  pass");
    String raiseError = code("def my_function():",
      "  raise NotImplementedError()",
      "  pass");
    PythonQuickFixVerifier.verify(check, codeWithIssue,
      addComment,
      raiseError);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue,
      "Insert placeholder comment",
      "Raise NotImplementedError()"
      );
  }

  @Test
  public void quick_fixes_for_method() {
    String codeWithIssue = code("class my_class:",
      "  def my_method():",
      "    pass");
    String addComment = code("class my_class:",
      "  def my_method():",
      "    # TODO document why this method is empty",
      "    pass");
    String raiseError = code("class my_class:",
      "  def my_method():",
      "    raise NotImplementedError()",
      "    pass");
    PythonQuickFixVerifier.verify(check, codeWithIssue, addComment, raiseError);
  }

  @Test
  public void quick_fixes_for_magic_binary_method() {
    String codeWithIssue = code("class my_class:",
      "  def __add__():",
      "    pass");
    String addComment = code("class my_class:",
      "  def __add__():",
      "    # TODO document why this method is empty",
      "    pass");
    String raiseError = code("class my_class:",
      "  def __add__():",
      "    return NotImplemented",
      "    pass");
    PythonQuickFixVerifier.verify(check, codeWithIssue, addComment, raiseError);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue,
      "Insert placeholder comment",
      "Return NotImplemented constant"
    );
  }

}
