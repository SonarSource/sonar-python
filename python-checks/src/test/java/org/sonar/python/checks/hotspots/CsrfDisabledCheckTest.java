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
package org.sonar.python.checks.hotspots;

import org.junit.jupiter.api.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class CsrfDisabledCheckTest {

  private static void testFile(String relPath) {
    String path = "src/test/resources/checks/hotspots/csrfDisabledCheck/" + relPath;
    PythonCheckVerifier.verify(path, new CsrfDisabledCheck());
  }

  @Test
  public void testMiddlewareArray() {
    testFile("django/settings.py");
  }

  @Test
  public void testCsrfExempt() {
    testFile("django/views.py");
  }

  @Test
  public void testWtfCsrfEnabledFalse() {
    testFile("flask/wtfCsrfEnabledFalse.py");
  }

  @Test
  public void testThreeWaysToDeactivateCsrfInFlaskForm() {
    testFile("flask/flaskform1.py");
  }

  @Test
  public void testGloballyMissingCSRFProtect(){
    testFile("flask/global.py");
  }

  @Test
  public void testExemptDecorators() {
    testFile("flask/flaskExempt.py");
    testFile("django/djangoExempt.py");
  }

  @Test
  public void testExemptAsFunction() {
    testFile("flask/exemptAsFunction.py");
  }

  @Test
  public void fixupTestsMoreRobustCSRFProtect() { testFile("flask/fixupTestsMoreRobustCSRFProtect.py"); }

  @Test
  public void fixupCsrfInGlobalScope() { testFile("flask/fixupCsrfInGlobalScope.py"); }

}
