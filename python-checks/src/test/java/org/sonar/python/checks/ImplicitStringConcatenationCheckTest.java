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

public class ImplicitStringConcatenationCheckTest {

  private static final PythonCheck check = new ImplicitStringConcatenationCheck();

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/implicitStringConcatenation.py", check);
  }

  @Test
  public void simple_expression_quickfix() {
    String codeWithIssue = "a = '1' '2'";
    String codeFixed1 = "a = '1' + '2'";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1);

    codeWithIssue = "a = ['1' '2']";
    codeFixed1 = "a = ['1', '2']";
    String codeFixed2 = "a = ['1' + '2']";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  public void complex_expression_quickfix() {
    String codeWithIssue = "def a():\n" +
      "    b = ['1' '2']\n";
    String codeFixed1 = "def a():\n" +
      "    b = ['1', '2']\n";
    String codeFixed2 = "def a():\n" +
      "    b = ['1' + '2']\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  public void concat_expression() {
    String codeWithIssue = "def a():\n" +
      "    b = ['1'+'2' '3'+'4']\n";
    String codeFixed1 = "def a():\n" +
      "    b = ['1'+'2', '3'+'4']\n";
    String codeFixed2 = "def a():\n" +
      "    b = ['1'+'2' + '3'+'4']\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  public void sets() {
    String codeWithIssue = "def a():\n" +
      "    {'1' '2', '3'}\n";
    String codeFixed1 = "def a():\n" +
      "    {'1', '2', '3'}\n";
    String codeFixed2 = "def a():\n" +
      "    {'1' + '2', '3'}\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  public void parameters() {
    String codeWithIssue = "def a():\n" +
      "    print('1' '2', '3')\n";
    String codeFixed1 = "def a():\n" +
      "    print('1', '2', '3')\n";
    String codeFixed2 = "def a():\n" +
      "    print('1' + '2', '3')\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

  @Test
  public void parameters2() {
    String codeWithIssue = "def a():\n" +
      "    foo('1' '2', '3')\n";
    String codeFixed1 = "def a():\n" +
      "    foo('1', '2', '3')\n";
    String codeFixed2 = "def a():\n" +
      "    foo('1' + '2', '3')\n";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed1, codeFixed2);
  }

}
