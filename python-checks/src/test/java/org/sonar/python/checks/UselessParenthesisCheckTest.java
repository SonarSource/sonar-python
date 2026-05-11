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
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.types.TypeShed;

import static org.junit.Assert.assertEquals;

class UselessParenthesisCheckTest {

  @AfterEach
  void reset_python_version() {
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void test() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_314));
    PythonCheckVerifier.verify("src/test/resources/checks/uselessParenthesis.py", new UselessParenthesisCheck());
  }
    
  @Test
  void test_older_python_version() {
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_313, PythonVersionUtils.Version.V_314));
    var issues313 = PythonCheckVerifier.issues("src/test/resources/checks/uselessParenthesis.py", new UselessParenthesisCheck());
    ProjectPythonVersion.setCurrentVersions(EnumSet.of(PythonVersionUtils.Version.V_314));
    var issues314 = PythonCheckVerifier.issues("src/test/resources/checks/uselessParenthesis.py", new UselessParenthesisCheck());
    assertEquals(issues314.size(), issues313.size() + 3);
  }

  @Test
  void quickFixTest() {
    PythonQuickFixVerifier.verify(new UselessParenthesisCheck(), "assert ((x < 2))", "assert (x < 2)");
    PythonQuickFixVerifier.verify(new UselessParenthesisCheck(),
      "for (x) in ((range(0, 3))):\n  pass",
      "for (x) in (range(0, 3)):\n  pass");
    PythonQuickFixVerifier.verifyQuickFixMessages(new UselessParenthesisCheck(),
      "for (x) in ((range(0, 3))):\n  pass", UselessParenthesisCheck.QUICK_FIX_MESSAGE);
  }

}
