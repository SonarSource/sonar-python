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

import org.junit.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class IdentityComparisonWithNewObjectCheckTest {
  @Test
  public void test() {
    PythonCheckVerifier.verify(
      "src/test/resources/checks/identityComparisonWithNewObjects.py",
      new IdentityComparisonWithNewObjectCheck());
  }

  @Test
  public void testIsReplacementQuickfix() {
    var check = new IdentityComparisonWithNewObjectCheck();
    String codeWithIssue = "def comprehensions(p):\n" +
      "  p is { x: x for x in range(10) }";
    String codeFixed = "def comprehensions(p):\n" +
      "  p == { x: x for x in range(10) }";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, IdentityComparisonWithNewObjectCheck.IS_QUICK_FIX_MESSAGE);
  }

  @Test
  public void testIsNotReplacementQuickfix() {
    var check = new IdentityComparisonWithNewObjectCheck();
    String codeWithIssue = "def comprehensions(p):\n" +
      "  p is not { x: x for x in range(10) }";
    String codeFixed = "def comprehensions(p):\n" +
      "  p != { x: x for x in range(10) }";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, IdentityComparisonWithNewObjectCheck.IS_NOT_QUICK_FIX_MESSAGE);
  }
}
