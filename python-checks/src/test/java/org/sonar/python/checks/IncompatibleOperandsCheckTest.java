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

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class IncompatibleOperandsCheckTest {

  @Test
  void test_comparison() {
    PythonCheckVerifier.verify("src/test/resources/checks/incompatibleOperands/comparison.py", new IncompatibleOperandsCheck());
  }

  @Test
  void test_arithmetic() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/incompatibleOperands/arithmetic.py",
      "src/test/resources/checks/classDecorators/importedDecorators.py"), new IncompatibleOperandsCheck());
  }

  @Test
  void instance_decorator_effects_on_unary_operations() {
    PythonCheckVerifier.verify(List.of(
      "src/test/resources/checks/classDecorators/instanceUnary.py",
      "src/test/resources/checks/classDecorators/importedDecorators.py"), new IncompatibleOperandsCheck());
  }

  @Test
  void project_function_marker_remains_conservative() {
    PythonCheckVerifier.verifyNoIssue(List.of(
      "src/test/resources/checks/classDecoratorFunctionImpostor/usage.py",
      "src/test/resources/checks/classDecoratorFunctionImpostor/typing.py"), new IncompatibleOperandsCheck());
  }
}
