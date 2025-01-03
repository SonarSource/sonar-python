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
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class IsCloseAbsTolCheckTest {

  private final PythonCheck check = new IsCloseAbsTolCheck();

  @Test
  void is_close_check() {
    PythonCheckVerifier.verify("src/test/resources/checks/isCloseAbsTol.py", check);
  }

  @Test
  void quickfix() {
    String noncompliant = "import math\n" +
      "def foo(a):\n" +
      "   math.isclose(a, 0)";

    String fixed = "import math\n" +
      "def foo(a):\n" +
      "   math.isclose(a, 0, abs_tol=1e-9)";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Add the \"abs_tol\" parameter.");
  }

  @Test
  void quickfix_with_assignement() {
    String noncompliant = "import math\n" +
      "def foo(a):\n" +
      "   b = 0\n" +
      "   math.isclose(a, b)";

    String fixed = "import math\n" +
      "def foo(a):\n" +
      "   b = 0\n" +
      "   math.isclose(a, b, abs_tol=1e-9)";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }

  @Test
  void quickfix_with_rel_tol() {
    String noncompliant = "import math\n" +
      "def foo(a):\n" +
      "   math.isclose(0, a, rel_tol=1e-8)";

    String fixed = "import math\n" +
      "def foo(a):\n" +
      "   math.isclose(0, a, rel_tol=1e-8, abs_tol=1e-9)";
    PythonQuickFixVerifier.verify(check, noncompliant, fixed);
  }
}
