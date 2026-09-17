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

class NonOctalPermissionModeCheckTest {

  private static final NonOctalPermissionModeCheck CHECK = new NonOctalPermissionModeCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/nonOctalPermissionMode.py", CHECK);
  }

  @Test
  void quick_fix_shell_style_octal_intent() {
    PythonQuickFixVerifier.verify(
      CHECK,
      """
        import os
        os.chmod("file.txt", 421)
        """,
      """
        import os
        os.chmod("file.txt", 0o645)
        """,
      """
        import os
        os.chmod("file.txt", 0o421)
        """);
    PythonQuickFixVerifier.verifyQuickFixMessages(
      CHECK,
      """
        import os
        os.chmod("file.txt", 421)
        """,
      "Convert to \"0o645\"",
      "Replace with \"0o421\"");
  }

  @Test
  void quick_fix_decimal_value_and_shell_style_when_both_apply() {
    PythonQuickFixVerifier.verify(
      CHECK,
      """
        import os
        os.chmod("file.txt", 511)
        """,
      """
        import os
        os.chmod("file.txt", 0o777)
        """,
      """
        import os
        os.chmod("file.txt", 0o511)
        """);
    PythonQuickFixVerifier.verifyQuickFixMessages(
      CHECK,
      """
        import os
        os.chmod("file.txt", 511)
        """,
      "Convert to \"0o777\"",
      "Replace with \"0o511\"");
  }

  @Test
  void quick_fix_decimal_value_and_shell_style_for_permission_mode() {
    PythonQuickFixVerifier.verify(
      CHECK,
      """
        import os
        os.chmod("file.txt", 420)
        """,
      """
        import os
        os.chmod("file.txt", 0o644)
        """,
      """
        import os
        os.chmod("file.txt", 0o420)
        """);
    PythonQuickFixVerifier.verifyQuickFixMessages(
      CHECK,
      """
        import os
        os.chmod("file.txt", 420)
        """,
      "Convert to \"0o644\"",
      "Replace with \"0o420\"");
  }

  @Test
  void quick_fix_short_shell_style_octal_intent() {
    PythonQuickFixVerifier.verify(
      CHECK,
      """
        import os
        os.umask(22)
        """,
      """
        import os
        os.umask(0o26)
        """,
      """
        import os
        os.umask(0o22)
        """);
  }

  @Test
  void quick_fix_decimal_value_only_when_digits_not_shell_style() {
    PythonQuickFixVerifier.verify(
      CHECK,
      """
        import os
        os.chmod("file.txt", 89)
        """,
      """
        import os
        os.chmod("file.txt", 0o131)
        """);
    PythonQuickFixVerifier.verifyQuickFixMessages(
      CHECK,
      """
        import os
        os.chmod("file.txt", 89)
        """,
      "Convert to \"0o131\"");
  }

  @Test
  void no_quick_fix_when_decimal_value_exceeds_unix_permission_range() {
    PythonQuickFixVerifier.verifyNoQuickFixes(
      CHECK,
      """
        import os
        os.chmod("file.txt", 755)
        """);
    PythonQuickFixVerifier.verifyNoQuickFixes(
      CHECK,
      """
        import os
        os.chmod("file.txt", 800)
        """);
    PythonQuickFixVerifier.verifyNoQuickFixes(
      CHECK,
      """
        import os
        os.chmod("file.txt", 33152)
        """);
  }
}
