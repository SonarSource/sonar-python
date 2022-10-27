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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.sonar.python.checks.utils.CodeTestUtils.code;

public class DuplicatedMethodImplementationCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/duplicatedMethodImplementationCheck.py", new DuplicatedMethodImplementationCheck());
  }

  @Test
  public void testQuickFixSimple() {
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
      "    return method()",
      "");

    PythonQuickFixVerifier.verify(new DuplicatedMethodImplementationCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixWithIfExpression() {
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
      "    return method_1()");

    PythonQuickFixVerifier.verify(new DuplicatedMethodImplementationCheck(), code, fixedCode);
  }

  @Test
  public void testQuickFixMessage() {
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

    PythonQuickFixVerifier.verifyQuickFixMessages(new DuplicatedMethodImplementationCheck(), code, "Call method_1 inside this function.");
  }
}
