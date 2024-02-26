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

class NumpyIsNanCheckTest {

  NumpyIsNanCheck check = new NumpyIsNanCheck();
  private static final String QUICK_FIX_MESSAGE_EQUALITY = "Replace this equality check with \"numpy.isnan()\".";
  private static final String QUICK_FIX_MESSAGE_INEQUALITY = "Replace this inequality check with \"not numpy.isnan()\".";

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/numpyIsNan.py", new NumpyIsNanCheck());
  }

  @Test
  void quickFixTestEqual() {

    final String nonCompliant1 = "import numpy as np\n" +
      "def foo(x):\n" +
      "    if x == np.nan: print(1)";

    final String nonCompliant2 = "import numpy as np\n" +
      "def foo(x):\n" +
      "    if np.nan == x: print(1)";

    final String compliant = "import numpy as np\n" +
      "def foo(x):\n" +
      "    if np.isnan(x): print(1)";

    performVerification(nonCompliant1, compliant, QUICK_FIX_MESSAGE_EQUALITY);
    performVerification(nonCompliant2, compliant, QUICK_FIX_MESSAGE_EQUALITY);
  }

  @Test
  void quickFixTestNotEqual() {

    final String nonCompliant1 = "import numpy as np\n" +
      "def foo(x):\n" +
      "    if x != np.nan: print(1)";

    final String nonCompliant2 = "import numpy as np\n" +
      "def foo(x):\n" +
      "    if np.nan != x: print(1)";

    final String compliant = "import numpy as np\n" +
      "def foo(x):\n" +
      "    if not np.isnan(x): print(1)";

    performVerification(nonCompliant1, compliant, QUICK_FIX_MESSAGE_INEQUALITY);
    performVerification(nonCompliant2, compliant, QUICK_FIX_MESSAGE_INEQUALITY);
  }

  private void performVerification(String nonCompliant, String compliant, String message) {
    PythonQuickFixVerifier.verify(check, nonCompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, nonCompliant, message);
  }

  @Test
  void quickFixTestQualifiedExpression() {

    final String nonCompliant = "import numpy as np\n" +
      "def foo(x):\n" +
      "    if np.max(2,5) == np.nan: print(1)";

    final String compliant = "import numpy as np\n" +
      "def foo(x):\n" +
      "    if np.isnan(np.max(2,5)): print(1)";

    performVerification(nonCompliant, compliant, QUICK_FIX_MESSAGE_EQUALITY);
  }

  @Test
  void noQuickFixTest() {
    // Here we offer no quick fixes, because we do not have a call to numpy.nan.
    final String compliant = "from numpy import nan\n" +
      "def foo(x):\n" +
      "    if x != nan: print(1)";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, compliant);
  }
}
