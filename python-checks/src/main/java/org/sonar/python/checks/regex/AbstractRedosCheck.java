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
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonarsource.analyzer.commons.regex.MatchType;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.finders.RedosFinder;

public abstract class AbstractRedosCheck extends AbstractRegexCheck {

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    MatchType matchType = RedosMatchTypeHelper.getMatchTypeFromCalledMethod(regexFunctionCall);
    new PythonRedosFinder().checkRegex(regexParseResult, matchType, this::addIssue);
  }

  protected abstract Optional<String> buildMessage(RedosFinder.BacktrackingType backtrackingType, boolean regexContainsBackReference);

  private class PythonRedosFinder extends RedosFinder {
    @Override
    protected Optional<String> message(RedosFinder.BacktrackingType backtrackingType, boolean regexContainsBackReference) {
      return buildMessage(backtrackingType, regexContainsBackReference);
    }
  }
}
