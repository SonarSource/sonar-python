/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.regex;

import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.regex.PythonRegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;
import org.sonarsource.analyzer.commons.regex.finders.VerboseRegexFinder;

@Rule(key = "S6353")
public class VerboseRegexCheck extends AbstractRegexCheck {

  private static final String ISSUE_MESSAGE_PATTERN = ".+syntax '(.+)' instead of.+";
  private static final Pattern issueMessagePattern = Pattern.compile(ISSUE_MESSAGE_PATTERN);
  public static final String QUICK_FIX_FORMAT = "Replace with \"%s\"";

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    new VerboseRegexFinder(this::addIssue).visit(regexParseResult);
  }

  @Override
  public PreciseIssue addIssue(RegexSyntaxElement regexTree, String message, @Nullable Integer cost, List<RegexIssueLocation> secondaries) {
    return Optional.of(super.addIssue(regexTree, message, cost, secondaries))
      .map(IssueWithQuickFix.class::cast)
      .map(issue -> {
        Matcher matcher = issueMessagePattern.matcher(message);
        String quickFixReplacement = matcher.replaceFirst("$1");

        IssueLocation issueLocation = PythonRegexIssueLocation.preciseLocation(regexTree, null);

        var textEdit = new PythonTextEdit(quickFixReplacement,
          issueLocation.startLine(),
          issueLocation.startLineOffset(),
          issueLocation.endLine(),
          issueLocation.endLineOffset());
        issue.addQuickFix(PythonQuickFix.newQuickFix(String.format(QUICK_FIX_FORMAT, quickFixReplacement), textEdit));
        return issue;
      })
      .orElse(null);
  }
}
