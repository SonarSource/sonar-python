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
package org.sonar.python.checks;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class NumpyWhereOneConditionCheckTest {

  NumpyWhereOneConditionCheck check = new NumpyWhereOneConditionCheck();

  @Test
  void test1() {
    PythonCheckVerifier.verify("src/test/resources/checks/numpyWhereOneCondition.py", check);
  }

  @Test
  void quickfix_test1() {

    final String quickFixMessage = "Replace numpy.where with numpy.nonzero";

    final String nonCompliant = "import numpy as np\n" +
      "def bigger_than_two():\n" +
      "  arr = np.array([1,2,3,4])\n" +
      "  result = np.where(arr > 2)";

    final String compliant = "import numpy as np\n" +
      "def bigger_than_two():\n" +
      "  arr = np.array([1,2,3,4])\n" +
      "  result = np.nonzero(arr > 2)";

    PythonQuickFixVerifier.verify(check, nonCompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, nonCompliant, quickFixMessage);
  }

  @Test
  void quickfix_test2() {

    // Here we don't offer a quickfix, because the imports of the functions can be complicated to track (nonzero could be unimported)

    final String compliant = "from numpy import array, where\n" +
      "def bigger_than_two():\n" +
      "  arr = array([1,2,3,4])\n" +
      "  result = where(arr > 2)";

    PythonQuickFixVerifier.verifyNoQuickFixes(check, compliant);
  }

}
