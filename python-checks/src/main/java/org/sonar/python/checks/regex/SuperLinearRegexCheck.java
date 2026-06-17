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
package org.sonar.python.checks.regex;


import java.util.Optional;
import org.sonar.check.Rule;
import org.sonarsource.analyzer.commons.regex.finders.RedosFinder;

@Rule(key = "S8786")
public class SuperLinearRegexCheck extends AbstractRedosCheck {

  private static final String MESSAGE = "Simplify this regular expression to reduce its runtime, as it has super-linear performance due to backtracking.";

  @Override
  protected Optional<String> buildMessage(RedosFinder.BacktrackingType backtrackingType, boolean regexContainsBackReference) {
    return switch (backtrackingType) {
      case ALWAYS_QUADRATIC -> Optional.of(MESSAGE);
      default -> Optional.empty();
    };
  }
}
