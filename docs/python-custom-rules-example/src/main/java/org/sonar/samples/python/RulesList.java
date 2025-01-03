/*
 * Copyright (C) 2011-2025 SonarSource SA - mailto:info AT sonarsource DOT com
 * This code is released under [MIT No Attribution](https://opensource.org/licenses/MIT-0) license.
 */
package org.sonar.samples.python;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.samples.python.checks.CustomPythonSubscriptionCheck;
import org.sonar.samples.python.checks.CustomPythonVisitorCheck;

public final class RulesList {

  private RulesList() {
  }

  public static List<Class<? extends PythonCheck>> getChecks() {
    return new ArrayList<>(Stream.concat(
      getPythonChecks().stream(),
      getPythonTestChecks().stream()
    ).toList());
  }

  /**
   * These rules are going to target MAIN code only
   */
  public static List<Class<? extends PythonCheck>> getPythonChecks() {
    return new ArrayList<>(List.of(
      CustomPythonSubscriptionCheck.class
    ));
  }

  /**
   * These rules are going to target TEST code only
   */
  public static List<Class<? extends PythonCheck>> getPythonTestChecks() {
    return new ArrayList<>(List.of(
      CustomPythonVisitorCheck.class
    ));
  }
}
