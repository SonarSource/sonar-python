/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
