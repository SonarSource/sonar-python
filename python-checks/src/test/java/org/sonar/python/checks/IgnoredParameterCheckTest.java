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

class IgnoredParameterCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/ignoredParameter.py", new IgnoredParameterCheck());
  }

  @Test
  void quickfix_assignment_statement() {
    var before = """
      def ignored_param(p):
          p = 42
          print(p)
      """;
    var after = """
      def ignored_param(p):
          p_value = 42
          print(p_value)
      """;

    PythonQuickFixVerifier.verify(new IgnoredParameterCheck(), before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(new IgnoredParameterCheck(), before, "Rename the reassigned value to 'p_value'");
  }

  @Test
  void quickfix_assignment_expression_with_name_collision() {
    var before = """
      def ignored_param(p):
          p_value = 0
          if (p := compute()):
              return p
          return p_value
      """;
    var after = """
      def ignored_param(p):
          p_value = 0
          if (p_value_1 := compute()):
              return p_value_1
          return p_value
      """;

    PythonQuickFixVerifier.verify(new IgnoredParameterCheck(), before, after);
  }
}
