/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

class FloatingPointEqualityCheckTest {

  private final FloatingPointEqualityCheck check = new FloatingPointEqualityCheck();
  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/floatingPointEquality.py", check);
  }

  @Test
  void quickfixEquality(){
    String noncompliant =
      """
      import math
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import math
      def foo(a,b):
          if math.isclose(a - 0.1, b, rel_tol=1e-09, abs_tol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"math.isclose()\".");
  }

  @Test
  void quickfixNoSpace(){
    String noncompliant =
      """
      import math
      def foo(a,b):
          if a - 0.1==b:
              ...""";
    String compliant =
      """
      import math
      def foo(a,b):
          if math.isclose(a - 0.1, b, rel_tol=1e-09, abs_tol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
  }
  
  @Test
  void quickfixInequality(){
    String noncompliant =
      """
      import math
      def foo(a,b):
          if a - 0.1 != b:
              ...""";
    String compliant =
      """
      import math
      def foo(a,b):
          if not math.isclose(a - 0.1, b, rel_tol=1e-09, abs_tol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"not math.isclose()\".");
  }

  @Test
  void quickfixRightOperand(){
    String noncompliant =
      """
      import math
      def foo(a,b):
          if a == .2 + b:
              ...""";
    String compliant =
      """
      import math
      def foo(a,b):
          if math.isclose(a, .2 + b, rel_tol=1e-09, abs_tol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
  }

  @Test
  void quickfixMathImport(){
    String noncompliant =
      """
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import math
      def foo(a,b):
          if math.isclose(a - 0.1, b, rel_tol=1e-09, abs_tol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
  }


  @Test
  void quickfixNumpyImport(){
    String noncompliant =
      """
      import numpy
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import numpy
      def foo(a,b):
          if numpy.isclose(a - 0.1, b, rtol=1e-09, atol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"numpy.isclose()\".");
  }

  @Test
  void quickfixMultipleImport(){
    String noncompliant =
      """
      import some_module, math
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import some_module, math
      def foo(a,b):
          if math.isclose(a - 0.1, b, rel_tol=1e-09, abs_tol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"math.isclose()\".");
  }

  @Test
  void quickfixPrioritizeNumpyOverMath(){
    String noncompliant =
      """
      import math
      import numpy as np
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import math
      import numpy as np
      def foo(a,b):
          if np.isclose(a - 0.1, b, rtol=1e-09, atol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"np.isclose()\".");
  }

  @Test
  void quickfixPrioritizeTorchOverMath(){
    String noncompliant =
      """
      import math, torch
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import math, torch
      def foo(a,b):
          if torch.isclose(a - 0.1, b, rtol=1e-09, atol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"torch.isclose()\".");
  }

  @Test
  void quickfixNumpyImportAlias(){
    String noncompliant =
      """
      import numpy as np
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import numpy as np
      def foo(a,b):
          if np.isclose(a - 0.1, b, rtol=1e-09, atol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"np.isclose()\".");
  }

  @Test
  void quickfixPyTorchImport(){
    String noncompliant =
      """
      import torch
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import torch
      def foo(a,b):
          if torch.isclose(a - 0.1, b, rtol=1e-09, atol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"torch.isclose()\".");
  }

  @Test
  void quickfixTakeFirstImport(){
    String noncompliant =
      """
      import torch
      import numpy
      def foo(a,b):
          if a - 0.1 == b:
              ...""";
    String compliant =
      """
      import torch
      import numpy
      def foo(a,b):
          if torch.isclose(a - 0.1, b, rtol=1e-09, atol=1e-09):
              ...""";
    PythonQuickFixVerifier.verify(check, noncompliant, compliant);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, noncompliant, "Replace with \"torch.isclose()\".");
  }
}
