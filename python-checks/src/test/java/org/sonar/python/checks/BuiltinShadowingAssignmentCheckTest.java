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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

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
        def inner():
          nonlocal int
          int = 43
          return int
        print(int)
        return int""";

    var after = """
      def my_function():
        _int = 42
        def inner():
          nonlocal _int
          _int = 43
          return _int
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

  @Test
  void noQuickFixWhenRenameWouldCaptureNestedReference() {
    PythonQuickFixVerifier.verifyNoQuickFixes(new BuiltinShadowingAssignmentCheck(), """
      _int = 1
      def outer():
        int = 42
        def inner():
          return int + _int
        return inner()
      """);
  }

  @Test
  void noQuickFixForSymbolContainingNestedGlobalUsages() {
    PythonQuickFixVerifier.verifyNoQuickFixes(new BuiltinShadowingAssignmentCheck(), """
      def outer():
        int = 1
        def inner():
          global int
          int = 2
        return int
      """);
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "[(int := x) for x in range(2)]",
    "{(int := x) for x in range(2)}",
    "{x: (int := x) for x in range(2)}",
    "list((int := x) for x in range(2))"
  })
  void noQuickFixForComprehensionWalrus(String comprehension) {
    PythonQuickFixVerifier.verifyNoQuickFixes(new BuiltinShadowingAssignmentCheck(), """
      def f():
        values = %s
        return int
      """.formatted(comprehension));
  }

  @Test
  void quickFixForOrdinaryWalrus() {
    PythonQuickFixVerifier.verify(new BuiltinShadowingAssignmentCheck(), """
      def f():
        if (int := value()):
          return int
      """, """
      def f():
        if (_int := value()):
          return _int
      """);
  }

  @Test
  void noQuickFixForOuterBindingWithComprehensionWalrus() {
    var issues = PythonCheckVerifier.issues("src/test/resources/checks/builtinShadowingComprehension.py", new BuiltinShadowingAssignmentCheck());
    assertThat(issues).hasSize(1).allSatisfy(issue -> assertThat(issue.quickFixes()).isEmpty());
  }

}
