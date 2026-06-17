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

class LocalVariableAndParameterNameConventionCheckTest {

  @Test
  void test() {
    LocalVariableAndParameterNameConventionCheck check = new LocalVariableAndParameterNameConventionCheck();
    check.format = "^[_a-z][a-z0-9_]+$";
    PythonCheckVerifier.verify("src/test/resources/checks/localVariableAndParameterNameIncompatibility.py", check);

  }

  @Test
  void quickfix_parameter() {
    var before = """
      def foo(inputPar):
          return inputPar + 1
      """;
    var after = """
      def foo(input_par):
          return input_par + 1
      """;

    PythonQuickFixVerifier.verify(new LocalVariableAndParameterNameConventionCheck(), before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(new LocalVariableAndParameterNameConventionCheck(), before, "Rename 'inputPar' to 'input_par'");
  }

  @Test
  void quickfix_local_variable() {
    var before = """
      def foo():
          localVar = 1
          return localVar
      """;
    var after = """
      def foo():
          local_var = 1
          return local_var
      """;

    PythonQuickFixVerifier.verify(new LocalVariableAndParameterNameConventionCheck(), before, after);
  }

}
