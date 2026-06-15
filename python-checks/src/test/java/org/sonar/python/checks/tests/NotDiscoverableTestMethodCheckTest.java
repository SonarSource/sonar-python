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
package org.sonar.python.checks.tests;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.assertj.core.api.Assertions.assertThat;

class NotDiscoverableTestMethodCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/tests/notDiscoverableTestMethod.py", new NotDiscoverableTestMethodCheck());
  }

  @Test
  void quick_fix_renames_method() {
    var check = new NotDiscoverableTestMethodCheck();
    String before = """
        import unittest

        class MyTest(unittest.TestCase):

            def helper(self):
                ...
        """;
    PythonQuickFixVerifier.verify(
      check,
      before,
      """
        import unittest

        class MyTest(unittest.TestCase):

            def test_helper(self):
                ...
        """);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, "Rename 'helper' to 'test_helper'");
  }

  @Test
  void no_quick_fix_when_discoverable_name_already_exists() {
    PythonQuickFixVerifier.verifyNoQuickFixes(
      new NotDiscoverableTestMethodCheck(),
      """
        import unittest

        class MyTest(unittest.TestCase):

            def helper(self):
                ...
            def test_helper(self):
                ...
        """);
  }

  @Test
  void test_scope() {
    assertThat(new NotDiscoverableTestMethodCheck().scope()).isEqualTo(PythonCheck.CheckScope.ALL);
  }
}
