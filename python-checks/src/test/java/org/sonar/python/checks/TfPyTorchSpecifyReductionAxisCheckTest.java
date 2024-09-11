/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class TfPyTorchSpecifyReductionAxisCheckTest {
  @Test
  void testTensorFlow() {
    PythonCheckVerifier.verify("src/test/resources/checks/tfSpecifyReductionAxis.py", new TfPyTorchSpecifyReductionAxisCheck());
  }

  @Test
  void testTensorFlowQuickFix() {
    PythonQuickFixVerifier.verify(new TfPyTorchSpecifyReductionAxisCheck(),
      """
        from tensorflow import math
        math.reduce_all(input)
        """,
      """
        from tensorflow import math
        math.reduce_all(input, axis=None)
        """);

    PythonQuickFixVerifier.verify(new TfPyTorchSpecifyReductionAxisCheck(),
      """
        from tensorflow import math
        math.reduce_all(input, keepdims=True)
        """,
      """
        from tensorflow import math
        math.reduce_all(input, keepdims=True, axis=None)
        """);

    PythonQuickFixVerifier.verifyNoQuickFixes(new TfPyTorchSpecifyReductionAxisCheck(),
        """
        from tensorflow import math
        math.reduce_all()
        """);
  }

  @Test
  void testPyTorch() {
    PythonCheckVerifier.verify("src/test/resources/checks/pyTorchSpecifyReductionDim.py", new TfPyTorchSpecifyReductionAxisCheck());
  }
}
