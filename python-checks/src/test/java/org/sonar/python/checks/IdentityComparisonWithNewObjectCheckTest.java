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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class IdentityComparisonWithNewObjectCheckTest {
  @Test
  void test() {
    PythonCheckVerifier.verify(
      "src/test/resources/checks/identityComparisonWithNewObjects.py",
      new IdentityComparisonWithNewObjectCheck());
  }

  @Test
  void testIsReplacementQuickfix() {
    var check = new IdentityComparisonWithNewObjectCheck();
    String codeWithIssue = "def comprehensions(p):\n" +
      "  p is { x: x for x in range(10) }";
    String codeFixed = "def comprehensions(p):\n" +
      "  p == { x: x for x in range(10) }";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, IdentityComparisonWithNewObjectCheck.IS_QUICK_FIX_MESSAGE);
  }

  @Test
  void testIsNotReplacementQuickfix() {
    var check = new IdentityComparisonWithNewObjectCheck();
    String codeWithIssue = "def comprehensions(p):\n" +
      "  p is not { x: x for x in range(10) }";
    String codeFixed = "def comprehensions(p):\n" +
      "  p != { x: x for x in range(10) }";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, IdentityComparisonWithNewObjectCheck.IS_NOT_QUICK_FIX_MESSAGE);
  }
}
