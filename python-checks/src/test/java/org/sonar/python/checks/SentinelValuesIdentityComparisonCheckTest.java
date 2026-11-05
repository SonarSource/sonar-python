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

import java.util.EnumSet;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.types.TypeShed;

class SentinelValuesIdentityComparisonCheckTest {

  @AfterEach
  void resetPythonVersion() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void test() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_315));
    PythonCheckVerifier.verify("src/test/resources/checks/sentinelValuesIdentityComparison.py", new SentinelValuesIdentityComparisonCheck());
  }

  @Test
  void typing_extensions_sentinel() {
    PythonCheckVerifier.verify("src/test/resources/checks/typingExtensionsSentinelValuesIdentityComparison.py", new SentinelValuesIdentityComparisonCheck());
  }

  @Test
  void equality_comparison_quick_fix() {
    var check = new SentinelValuesIdentityComparisonCheck();
    String codeWithIssue = """
      from typing_extensions import Sentinel
      missing = Sentinel("MISSING")
      if value == missing:
          pass
      """;
    String fixedCode = """
      from typing_extensions import Sentinel
      missing = Sentinel("MISSING")
      if value is missing:
          pass
      """;
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, SentinelValuesIdentityComparisonCheck.IS_QUICK_FIX_MESSAGE);
  }

  @Test
  void equality_comparison_without_whitespace_quick_fix() {
    var check = new SentinelValuesIdentityComparisonCheck();
    String codeWithIssue = """
      from typing_extensions import Sentinel
      missing = Sentinel("MISSING")
      if value==missing:
          pass
      """;
    String fixedCode = """
      from typing_extensions import Sentinel
      missing = Sentinel("MISSING")
      if value is missing:
          pass
      """;
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
  }

  @Test
  void inequality_comparison_quick_fix() {
    var check = new SentinelValuesIdentityComparisonCheck();
    String codeWithIssue = """
      from typing_extensions import Sentinel
      missing = Sentinel("MISSING")
      if value != missing:
          pass
      """;
    String fixedCode = """
      from typing_extensions import Sentinel
      missing = Sentinel("MISSING")
      if value is not missing:
          pass
      """;
    PythonQuickFixVerifier.verify(check, codeWithIssue, fixedCode);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, codeWithIssue, SentinelValuesIdentityComparisonCheck.IS_NOT_QUICK_FIX_MESSAGE);
  }
}
