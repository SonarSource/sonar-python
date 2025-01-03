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

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class BooleanCheckNotInvertedCheckTest {

  private final PythonCheck check = new BooleanCheckNotInvertedCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/booleanCheckNotInverted.py", check);
  }

  @Test
  void operatorStringTest() {
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString("==")).isEqualTo("!=");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString("!=")).isEqualTo("==");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString("<")).isEqualTo(">=");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString("<=")).isEqualTo(">");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString(">")).isEqualTo("<=");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString(">=")).isEqualTo("<");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString("is")).isEqualTo("is not");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString("is not")).isEqualTo("is");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString("in")).isEqualTo("not in");
    assertThat(BooleanCheckNotInvertedCheck.oppositeOperatorString("not in")).isEqualTo("in");

    Assertions.assertThatThrownBy(() -> BooleanCheckNotInvertedCheck.oppositeOperatorString("-")).isInstanceOf(IllegalArgumentException.class);
    Assertions.assertThatThrownBy(() -> BooleanCheckNotInvertedCheck.oppositeOperatorString("*")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void test_quickfix() {
    String codeWithIssue = "a = not(b == c)";
    String codeFixed = "a = b != c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not b != c";
    codeFixed = "a = b == c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b != c)";
    codeFixed = "a = b == c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b < c)";
    codeFixed = "a = b >= c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b is c)";
    codeFixed = "a = b is not c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b is not c)";
    codeFixed = "a = b is c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b in c)";
    codeFixed = "a = b not in c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b not in c)";
    codeFixed = "a = b in c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void not_in_if() {
    String codeWithIssue = "" +
      "def func():\n" +
      "    if not a == 2:\n" +
      "        b = 10\n" +
      "    return \"item1\" \"item2\"";
    String codeFixed = "" +
      "def func():\n" +
      "    if a != 2:\n" +
      "        b = 10\n" +
      "    return \"item1\" \"item2\"";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "" +
      "def func():\n" +
      "    if not a == 2 and b == 9:\n" +
      "        b = 10\n" +
      "    return \"item1\" \"item2\"";
    codeFixed = "" +
      "def func():\n" +
      "    if a != 2 and b == 9:\n" +
      "        b = 10\n" +
      "    return \"item1\" \"item2\"";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "" +
      "def func():\n" +
      "    if a != 2 and not b == 9:\n" +
      "        b = 10\n" +
      "    return \"item1\" \"item2\"";
    codeFixed = "" +
      "def func():\n" +
      "    if a != 2 and b != 9:\n" +
      "        b = 10\n" +
      "    return \"item1\" \"item2\"";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void not_parentheses() {
    String codeWithIssue = "a = not ((((((b > c))))))";
    String codeFixed = "a = b <= c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (not (b == c))";
    codeFixed = "a = not (b != c)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (a is (not b))";
    codeFixed = "a = a is not (not b)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (1 == b == c) == 2";
    codeFixed = "a = (1 == b == c) != 2";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (1 == b == c) == (d == 2)";
    codeFixed = "a = (1 == b == c) != (d == 2)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not ((1 == b == c) and a) == (d == 2)";
    codeFixed = "a = ((1 == b == c) and a) != (d == 2)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b < foo())";
    codeFixed = "a = b >= foo()";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void expression_list() {
    String codeWithIssue = "a = not [] < (c,\n" +
      "  d)";
    String codeFixed = "a = [] >= (c,\n" +
      "  d)";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not [a, (a,b)] is c";
    codeFixed = "a = [a, (a,b)] is not c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (foo((1)) < c)";
    codeFixed = "a = foo((1)) >= c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (foo(()) < c)";
    codeFixed = "a = foo(()) >= c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not (() in c)";
    codeFixed = "a = () not in c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not ((1,) in c)";
    codeFixed = "a = (1,) not in c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

    codeWithIssue = "a = not ((1,2,3,4) in c)";
    codeFixed = "a = (1,2,3,4) not in c";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void fstring() {
    String codeWithIssue = "x = not (a == f'foo${b}')";
    String codeFixed = "x = a != f'foo${b}'";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);

  }
  @Test
  void brackets(){
    String codeWithIssue = "x = not ( ham[1] in a)";
    String codeFixed = "x = ham[1] not in a";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }

  @Test
  void notInQuickFix(){
    String codeWithIssue = "def foo(key, mapping):\n" +
      "    if not key in mapping.keys():\n" +
      "        pass";
    String codeFixed = "def foo(key, mapping):\n" +
      "    if key not in mapping.keys():\n" +
      "        pass";
    PythonQuickFixVerifier.verify(check, codeWithIssue, codeFixed);
  }
}
