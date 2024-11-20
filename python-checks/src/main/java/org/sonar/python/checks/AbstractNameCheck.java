/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.checks;

import java.util.regex.Pattern;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;

public abstract class AbstractNameCheck extends PythonSubscriptionCheck {

  private Pattern pattern = null;

  protected Pattern pattern() {
    if (pattern == null) {
      pattern = Pattern.compile(format());
    }
    return pattern;
  }

  protected abstract String format();

}
