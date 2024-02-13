/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
import org.junit.Assert;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class StringLiteralDuplicationCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/stringLiteralDuplication.py", new StringLiteralDuplicationCheck());
  }

  @Test
  void test_no_issue_on_test_code() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/test_stringLiteralDuplication.py", new StringLiteralDuplicationCheck());
  }

  @Test
  void test_custom_pattern() {
    StringLiteralDuplicationCheck check = new StringLiteralDuplicationCheck();
    check.customExclusionRegex = "[a\\s]+";
    PythonCheckVerifier.verify("src/test/resources/checks/stringLiteralDuplicationCustom.py", check);
  }

  @Test
  void test_invalid_custom_pattern() {
    StringLiteralDuplicationCheck check = new StringLiteralDuplicationCheck();
    check.customExclusionRegex = "a+*(";
    IllegalStateException e = Assert.assertThrows(IllegalStateException.class,
      () -> PythonCheckVerifier.verify("src/test/resources/checks/stringLiteralDuplicationCustom.py", check));
    Assertions.assertThat(e.getMessage()).isEqualTo("Unable to compile regular expression: a+*(");

  }
}
