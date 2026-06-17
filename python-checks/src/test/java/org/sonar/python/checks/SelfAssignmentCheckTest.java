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

class SelfAssignmentCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/selfAssignment.py", new SelfAssignmentCheck());
  }

  @Test
  void quickfix_assignment_statement() {
    var before = """
      def foo(x):
          x = x
          return x
      """;
    var after = """
      def foo(x):
          return x
      """;

    PythonQuickFixVerifier.verify(new SelfAssignmentCheck(), before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(new SelfAssignmentCheck(), before, "Remove the self-assignment");
  }

  @Test
  void quickfix_assignment_expression() {
    var before = """
      def foo(x):
          if (x := x):
              return x
      """;
    var after = """
      def foo(x):
          if x:
              return x
      """;

    PythonQuickFixVerifier.verify(new SelfAssignmentCheck(), before, after);
  }

}
