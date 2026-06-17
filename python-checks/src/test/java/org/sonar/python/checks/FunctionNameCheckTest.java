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

class FunctionNameCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/functionName.py", new FunctionNameCheck());
  }

  @Test
  void quickfix() {
    var before = """
      def Badly_Named_Function():
          return 1
      
      print(Badly_Named_Function())
      """;
    var after = """
      def badly_named_function():
          return 1
      
      print(badly_named_function())
      """;

    PythonQuickFixVerifier.verify(new FunctionNameCheck(), before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(new FunctionNameCheck(), before, "Rename 'Badly_Named_Function' to 'badly_named_function'");
  }

}
