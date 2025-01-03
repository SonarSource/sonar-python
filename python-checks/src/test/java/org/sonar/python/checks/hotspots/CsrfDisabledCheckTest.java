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
package org.sonar.python.checks.hotspots;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class CsrfDisabledCheckTest {
  private static void testFile(String relPath) {
    testFile(List.of(relPath));
  }

  private static void testFile(List<String> relPaths) {
    List<String> absolutePaths = relPaths.stream().map(path -> "src/test/resources/checks/hotspots/csrfDisabledCheck/" + path).toList();
    PythonCheckVerifier.verify(absolutePaths, new CsrfDisabledCheck());
  }

  @Test
  void testMiddlewareArray() {
    testFile("django/settings.py");
  }

  @Test
  void testCsrfExempt() {
    testFile("django/views.py");
  }

  @Test
  void testWtfCsrfEnabledFalse() {
    testFile("flask/wtfCsrfEnabledFalse.py");
  }

  @Test
  void testThreeWaysToDeactivateCsrfInFlaskForm() {
    testFile("flask/flaskform1.py");
  }

  @Test
  void testGloballyMissingCSRFProtect(){
    testFile("flask/global.py");
  }

  @Test
  void testExemptDecorators() {
    testFile(List.of("flask/flaskExempt.py", "flask/exportedCsrfProtect.py"));
    testFile("django/djangoExempt.py");
  }

  @Test
  void testExemptAsFunction() {
    testFile("flask/exemptAsFunction.py");
  }

  @Test
  void fixupTestsMoreRobustCSRFProtect() { testFile("flask/fixupTestsMoreRobustCSRFProtect.py"); }

  @Test
  void fixupCsrfInGlobalScope() { testFile("flask/fixupCsrfInGlobalScope.py"); }

}
