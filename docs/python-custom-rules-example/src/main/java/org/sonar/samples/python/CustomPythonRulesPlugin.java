/*
 * Copyright (C) 2011-2025 SonarSource SA - mailto:info AT sonarsource DOT com
 * This code is released under [MIT No Attribution](https://opensource.org/licenses/MIT-0) license.
 */
package org.sonar.samples.python;

import org.sonar.api.Plugin;

public class CustomPythonRulesPlugin implements Plugin {

  @Override
  public void define(Context context) {
    context.addExtension(CustomPythonRuleRepository.class);
  }

}
