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

import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.python.regex.PythonRegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;
import org.sonarsource.analyzer.commons.regex.finders.SingleCharCharacterClassFinder;

@Rule(key = "S6397")
public class SingleCharCharacterClassCheck extends AbstractRegexCheck {
  public static final String QUICK_FIX_MESSAGE = "Replace this character class with the character itself";

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    Optional.ofNullable(regexFunctionCall.calleeSymbol())
      .flatMap(symbol -> Optional.ofNullable(symbol.fullyQualifiedName()))
      .filter(fqn -> lookedUpFunctions().containsKey(fqn))
      .filter(fqn -> !regexParseResult.getResult().activeFlags().contains(Pattern.COMMENTS))
      .ifPresent(fqn -> new SingleCharCharacterClassFinder(this::addIssue).visit(regexParseResult));
  }

  @Override
  public PreciseIssue addIssue(RegexSyntaxElement regexTree, String message, @Nullable Integer cost, List<RegexIssueLocation> secondaries) {
    var issue = super.addIssue(regexTree, message, cost, secondaries);
    var quickFixReplacement = regexTree.getText();
    var issueLocation = PythonRegexIssueLocation.preciseLocation(regexTree, null);
    var textEdit = new PythonTextEdit(quickFixReplacement,
      issueLocation.startLine(), issueLocation.startLineOffset() - 1,
      issueLocation.endLine(), issueLocation.endLineOffset() + 1);
    issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE, textEdit));
    return issue;
  }
}
