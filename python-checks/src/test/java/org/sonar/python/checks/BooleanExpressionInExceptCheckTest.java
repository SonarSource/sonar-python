/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

class BooleanExpressionInExceptCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/booleanExpressionInExcept.py", new BooleanExpressionInExceptCheck());
  }
  @Test
  void quickFixTest() {
    var before = """
      try:
          foo()
      except ValueError or TypeError and SomeError:
          pass""";

    var after = """
      try:
          foo()
      except (ValueError, TypeError, SomeError):
          pass""";
    verifyQuickFix(before, after);
  }

  @Test
  void wrappedInParenthesisQuickFixTest() {
    var before = """
      try:
          foo()
      except (ValueError or TypeError and SomeError):
          pass""";

    var after = """
      try:
          foo()
      except (ValueError, TypeError, SomeError):
          pass""";
    verifyQuickFix(before, after);
  }

  @Test
  void nestedQuickFixTest() {
    var before = """
      try:
          foo()
      except ((ValueError or TypeError) and pkg.cstm.SomeError):
          pass""";

    var after = """
      try:
          foo()
      except (ValueError, TypeError, pkg.cstm.SomeError):
          pass""";
    verifyQuickFix(before, after);
  }

  @Test
  void noQuickFixTest() {
    var before = """
      try:
          foo()
      except (ValueError or pkg.cstm.SomeError()):
          pass""";

    PythonQuickFixVerifier.verifyNoQuickFixes(new BooleanExpressionInExceptCheck(), before);
  }

  private void verifyQuickFix(String before, String after) {
    var check = new BooleanExpressionInExceptCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, BooleanExpressionInExceptCheck.QUICK_FIX_MESSAGE);
  }

}
