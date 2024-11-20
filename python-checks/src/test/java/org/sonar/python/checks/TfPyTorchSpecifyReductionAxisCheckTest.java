/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

class TfPyTorchSpecifyReductionAxisCheckTest {
  private static final TfPyTorchSpecifyReductionAxisCheck CHECK_OBJECT = new TfPyTorchSpecifyReductionAxisCheck();

  @Test
  void testTensorFlow() {
    PythonCheckVerifier.verify("src/test/resources/checks/tfSpecifyReductionAxis.py", CHECK_OBJECT);
  }

  @Test
  void testTensorFlowQuickFix() {
    PythonQuickFixVerifier.verify(CHECK_OBJECT,
      """
        from tensorflow import math
        math.reduce_all(input)
        """,
      """
        from tensorflow import math
        math.reduce_all(input, axis=None)
        """);

    PythonQuickFixVerifier.verify(CHECK_OBJECT,
      """
        from tensorflow import math
        math.reduce_all(input, keepdims=True)
        """,
      """
        from tensorflow import math
        math.reduce_all(input, keepdims=True, axis=None)
        """);

    PythonQuickFixVerifier.verifyNoQuickFixes(CHECK_OBJECT,
        """
        from tensorflow import math
        math.reduce_all()
        """);
  }

  @Test
  void testPyTorch() {
    PythonCheckVerifier.verify("src/test/resources/checks/pyTorchSpecifyReductionDim.py", CHECK_OBJECT);
  }
}
