package org.sonar.python.checks.regex;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.finders.ImpossibleBoundaryFinder;

@Rule(key = "S5996")
public class ImpossibleBoundariesCheck extends AbstractRegexCheck {

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    new ImpossibleBoundaryFinder(this::addIssue).visit(regexParseResult);
  }
}
