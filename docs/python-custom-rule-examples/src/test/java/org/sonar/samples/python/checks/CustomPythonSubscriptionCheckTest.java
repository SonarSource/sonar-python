/*
 * Copyright (C) 2011-2024 SonarSource SA - mailto:info AT sonarsource DOT com
 * This code is released under [MIT No Attribution](https://opensource.org/licenses/MIT-0) license.
 */
package org.sonar.samples.python.checks;

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class CustomPythonSubscriptionCheckTest {
  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/customPythonSubscriptionCheck.py", new CustomPythonSubscriptionCheck());
  }
}
