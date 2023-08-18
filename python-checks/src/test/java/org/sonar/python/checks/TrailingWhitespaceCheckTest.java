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

class TrailingWhitespaceCheckTest {

  final PythonCheck check = new TrailingWhitespaceCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/trailingWhitespace.py", check);
  }

  @Test
  void single_whitespace_at_end_of_line() {
    String codeWithIssue = code(
      "print(1) ",
      "print(2)");
    String fixedCode = code(
      "print(1)",
      "print(2)"
    );
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void multiple_whitespace_at_end_of_line() {
    String codeWithIssue = code(
      "print(1)   ",
      "print(2)");
    String fixedCode = code(
      "print(1)",
      "print(2)"
    );
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void single_whitespace_as_line() {
    String codeWithIssue = code(
      " ",
      "print(1)");
    String fixedCode = code(
      "",
      "print(1)"
    );
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void multiple_whitespace_as_line() {
    String codeWithIssue = code(
      "   ",
      "print(1)");
    String fixedCode = code(
      "",
      "print(1)"
    );
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void single_whitespace_at_end_of_file() {
    String codeWithIssue = code(
      "print(1) ");
    String fixedCode = code(
      "print(1)"
    );
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

}
