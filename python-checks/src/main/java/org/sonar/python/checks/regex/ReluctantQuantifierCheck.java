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
package org.sonar.python.checks.regex;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonarsource.analyzer.commons.regex.RegexIssueReporter;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.Quantifier;
import org.sonarsource.analyzer.commons.regex.finders.ReluctantQuantifierFinder;

@Rule(key = "S5857")
public class ReluctantQuantifierCheck extends AbstractRegexCheck {

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    new PythonReluctantQuantifierFinder(this::addIssue).visit(regexParseResult);
  }

  private static class PythonReluctantQuantifierFinder extends ReluctantQuantifierFinder {

    public PythonReluctantQuantifierFinder(RegexIssueReporter.ElementIssue regexElementIssueReporter) {
      super(regexElementIssueReporter);
    }

    @Override
    protected String makePossessiveOrGreedy(Quantifier quantifier, boolean possessive) {
      return super.makePossessiveOrGreedy(quantifier, false);
    }
  }
}

