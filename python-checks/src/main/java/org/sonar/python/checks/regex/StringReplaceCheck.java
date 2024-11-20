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

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.python.regex.PythonRegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.RegexTree;
import org.sonarsource.analyzer.commons.regex.ast.SequenceTree;

@Rule(key = "S5361")
public class StringReplaceCheck extends AbstractRegexCheck {

  private static final String MESSAGE = "Replace this \"re.sub()\" call by a \"str.replace()\" function call.";
  private static final String SECONDARY_MESSAGE = "Expression without regular expression features.";

  @Override
  protected Map<String, Integer> lookedUpFunctions() {
    Map<String, Integer> result = new HashMap<>();
    result.put("re.sub", 4);
    return result;
  }

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression callExpression) {
    RegexTree regex = regexParseResult.getResult();
    if (regexParseResult.hasSyntaxErrors() || regex.activeFlags().contains(Pattern.CASE_INSENSITIVE)) {
      return;
    }

    if (isPlainString(regex)) {
      regexContext.addIssue(callExpression.callee(), MESSAGE)
        .secondary(PythonRegexIssueLocation.preciseLocation(regex, SECONDARY_MESSAGE));
    }
  }

  private static boolean isPlainString(RegexTree regex) {
    return regex.is(RegexTree.Kind.CHARACTER)
      || (regex.is(RegexTree.Kind.SEQUENCE)
      && !((SequenceTree) regex).getItems().isEmpty()
      && ((SequenceTree) regex).getItems().stream().allMatch(item -> item.is(RegexTree.Kind.CHARACTER)));
  }

}
