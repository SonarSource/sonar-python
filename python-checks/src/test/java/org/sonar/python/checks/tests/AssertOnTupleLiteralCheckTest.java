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
package org.sonar.python.checks.tests;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class AssertOnTupleLiteralCheckTest {
  
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/assertOnTupleLiteral.py", new AssertOnTupleLiteralCheck());
  }

  @Test
  void quickFixTest() {
    PythonQuickFixVerifier.verifyQuickFixMessages(new AssertOnTupleLiteralCheck(), "assert (foo(),)", "Remove parentheses");
    PythonQuickFixVerifier.verify(new AssertOnTupleLiteralCheck(), "assert (foo(),)", "assert foo()");
    PythonQuickFixVerifier.verifyNoQuickFixes(new AssertOnTupleLiteralCheck(), "assert (foo(),b,)");
    PythonQuickFixVerifier.verifyNoQuickFixes(new AssertOnTupleLiteralCheck(), "assert (foo(),b)");
    PythonQuickFixVerifier.verifyNoQuickFixes(new AssertOnTupleLiteralCheck(), "assert ()");
  }

  @Test
  void test_scope() {
    assertThat(new AssertOnTupleLiteralCheck().scope()).isEqualTo(PythonCheck.CheckScope.ALL);
  }
}
