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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class FloatingPointEqualityCheckTest {

  private final FloatingPointEqualityCheck check = new FloatingPointEqualityCheck();
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/floatingPointEquality.py", check);
  }

  @Test
  void quickfixEquality(){
    String noncompliant =
      "def foo(a,b):\n" +
      "    if a - 0.1 == b:\n" +
      "        ...";
    String compliant =
      "def foo(a,b):\n" +
      "    if math.isclose(a - 0.1, b, rel_tol=1e-09, abs_tol=1e-09):\n" +
      "        ...";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with math.isclose().");
  }

  @Test
  void quickfixNoSpace(){
    String noncompliant =
      "def foo(a,b):\n" +
      "    if a - 0.1==b:\n" +
      "        ...";
    String compliant =
      "def foo(a,b):\n" +
      "    if math.isclose(a - 0.1, b, rel_tol=1e-09, abs_tol=1e-09):\n" +
      "        ...";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
  }
  
  @Test
  void quickfixInequality(){
    String noncompliant =
      "def foo(a,b):\n" +
      "    if a - 0.1 != b:\n" +
      "        ...";
    String compliant =
      "def foo(a,b):\n" +
      "    if !math.isclose(a - 0.1, b, rel_tol=1e-09, abs_tol=1e-09):\n" +
      "        ...";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with !math.isclose().");
  }

  @Test
  void quickfixRightOperand(){
    String noncompliant =
      "def foo(a,b):\n" +
      "    if a == .2 + b:\n" +
      "        ...";
    String compliant =
      "def foo(a,b):\n" +
      "    if math.isclose(a, .2 + b, rel_tol=1e-09, abs_tol=1e-09):\n" +
      "        ...";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
  }
}
