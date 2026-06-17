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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class UnreachableExceptCheckTest {

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/unreachableExcept.py", new UnreachableExceptCheck());
  }

  @Test
  void quickfix_remove_duplicate_except_clause() {
    var before = """
      def foo():
          try:
              work()
          except ValueError:
              handle_value()
          except ValueError:
              fallback()
      """;
    var after = """
      def foo():
          try:
              work()
          except ValueError:
              handle_value()
      """;

    PythonQuickFixVerifier.verify(new UnreachableExceptCheck(), before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(new UnreachableExceptCheck(), before, "Remove unreachable except clause");
  }

  @Test
  void quickfix_remove_duplicate_type_from_tuple() {
    var before = """
      def foo():
          try:
              work()
          except (ValueError, TypeError):
              handle_value()
          except (ValueError, RuntimeError):
              fallback()
      """;
    var after = """
      def foo():
          try:
              work()
          except (ValueError, TypeError):
              handle_value()
          except RuntimeError:
              fallback()
      """;

    PythonQuickFixVerifier.verify(new UnreachableExceptCheck(), before, after);
  }

}
