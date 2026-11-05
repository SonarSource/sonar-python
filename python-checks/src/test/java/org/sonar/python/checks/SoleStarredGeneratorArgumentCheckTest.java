/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

class SoleStarredGeneratorArgumentCheckTest {

  private static final SoleStarredGeneratorArgumentCheck check = new SoleStarredGeneratorArgumentCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/soleStarredGeneratorArgument.py", check);
  }

  @Test
  void quickFix() {
    PythonQuickFixVerifier.verify(check, "f(*x for x in groups)", "f((*x for x in groups))");
  }

  @Test
  void multilineQuickFix() {
    PythonQuickFixVerifier.verify(check, """
      f(
        *x for x in groups
      )""", """
      f(
        (*x for x in groups)
      )""");
  }
}
