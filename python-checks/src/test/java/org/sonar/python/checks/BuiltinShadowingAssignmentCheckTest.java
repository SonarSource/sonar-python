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

class BuiltinShadowingAssignmentCheckTest {

  @Test
  void testVariableShadowing() {
    PythonCheckVerifier.verify("src/test/resources/checks/builtinShadowing.py", new BuiltinShadowingAssignmentCheck());
  }

  @Test
  void quickFixTest() {
    var before = """
      def my_function():
        int = 42
        print(int)
        return int""";

    var after = """
      def my_function():
        _int = 42
        print(_int)
        return _int""";
    var check = new BuiltinShadowingAssignmentCheck();
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, String.format(BuiltinShadowingAssignmentCheck.QUICK_FIX_MESSAGE_FORMAT, "int"));
  }

  @Test
  void noQuickFixTest() {
    var check = new BuiltinShadowingAssignmentCheck();

    var before = """
      def my_function():
        _int = 22
        int = 42
        print(int)
        return int""";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);

    before = """
      def my_function(_int = 22):
        int = 42
        print(int)
        return int""";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);

    before = """
      def my_function((_int, b)):
        int = 42
        print(int)
        return int""";
    PythonQuickFixVerifier.verifyNoQuickFixes(check, before);
  }

}
