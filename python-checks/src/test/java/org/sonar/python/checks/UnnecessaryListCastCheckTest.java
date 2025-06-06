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

class UnnecessaryListCastCheckTest {

  private static final UnnecessaryListCastCheck check = new UnnecessaryListCastCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/unnecessaryListCast.py", check);
  }

  @Test
  void test_quick_fix() {
    String codeWithIssue = """
    for x in list([1,2,5]):
        ...
    """;
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, "Remove the \"list\" call");

    String correctCode = """
    for x in [1,2,5]:
        ...
    """;

    PythonQuickFixVerifier.verify(check, codeWithIssue, correctCode);
  }

  @Test
  void test_quick_fix_comprehension() {
    String codeWithIssue = "{i for i in list([1,2,3])}";
    String correctCode = "{i for i in [1,2,3]}";

    PythonQuickFixVerifier.verify(check, codeWithIssue, correctCode);
  }

  @Test
  void test_no_quick_fix_on_multiline() {
    String codeWithIssue = """
    for x in list(
        [
        1,
        2,
        5]):
        ...
    """;
    PythonQuickFixVerifier.verifyNoQuickFixes(check, codeWithIssue);
  }
}
