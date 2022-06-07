/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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

import org.assertj.core.api.Assertions;
import org.junit.Test;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

public class BooleanCheckNotInvertedCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/booleanCheckNotInverted.py", new BooleanCheckNotInvertedCheck());
  }

  @Test
  public void operatorStringTest() {
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
  public void test_quickfix() {
    String codeWithIssue = "a = not(b == c)";
    String codeFixed = "a = b != c";
    PythonQuickFixVerifier.verify(new BooleanCheckNotInvertedCheck(), codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b != c)";
    codeFixed = "a = b == c";
    PythonQuickFixVerifier.verify(new BooleanCheckNotInvertedCheck(), codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b < c)";
    codeFixed = "a = b >= c";
    PythonQuickFixVerifier.verify(new BooleanCheckNotInvertedCheck(), codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b is c)";
    codeFixed = "a = b is not c";
    PythonQuickFixVerifier.verify(new BooleanCheckNotInvertedCheck(), codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b is not c)";
    codeFixed = "a = b is c";
    PythonQuickFixVerifier.verify(new BooleanCheckNotInvertedCheck(), codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b in c)";
    codeFixed = "a = b not in c";
    PythonQuickFixVerifier.verify(new BooleanCheckNotInvertedCheck(), codeWithIssue, codeFixed);

    codeWithIssue = "a = not (b not in c)";
    codeFixed = "a = b in c";
    PythonQuickFixVerifier.verify(new BooleanCheckNotInvertedCheck(), codeWithIssue, codeFixed);
  }
}
