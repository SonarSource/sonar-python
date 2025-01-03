/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.python.regex.PythonRegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.NonCapturingGroupTree;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;
import org.sonarsource.analyzer.commons.regex.finders.UnquantifiedNonCapturingGroupFinder;

@Rule(key = "S6395")
public class UnquantifiedNonCapturingGroupCheck extends AbstractRegexCheck {
  public static final String QUICK_FIX_MESSAGE = "Unwrap subpattern";

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    new UnquantifiedNonCapturingGroupFinder(this::addIssue).visit(regexParseResult);
  }

  @Override
  public PreciseIssue addIssue(RegexSyntaxElement regexTree, String message, @Nullable Integer cost, List<RegexIssueLocation> secondaries) {
    var issue = super.addIssue(regexTree, message, cost, secondaries);

    Optional.of(regexTree)
      .filter(NonCapturingGroupTree.class::isInstance)
      .map(NonCapturingGroupTree.class::cast)
      .filter(group -> Objects.nonNull(group.getElement()))
      .map(group -> {
        var quickFixReplacement = group.getElement().getText();
        var issueLocation = PythonRegexIssueLocation.preciseLocation(group, null);
        var textEdit = new PythonTextEdit(quickFixReplacement,
          issueLocation.startLine(),
          issueLocation.startLineOffset(),
          issueLocation.endLine(),
          issueLocation.endLineOffset());
        return PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE, textEdit);
      }).ifPresent(issue::addQuickFix);

    return issue;
  }
}
