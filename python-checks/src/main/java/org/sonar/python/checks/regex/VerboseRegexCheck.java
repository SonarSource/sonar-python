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

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.python.regex.PythonRegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexIssueLocation;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.CharacterRangeTree;
import org.sonarsource.analyzer.commons.regex.ast.Quantifier;
import org.sonarsource.analyzer.commons.regex.ast.RegexBaseVisitor;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;
import org.sonarsource.analyzer.commons.regex.ast.RegexTree;
import org.sonarsource.analyzer.commons.regex.ast.RepetitionTree;
import org.sonarsource.analyzer.commons.regex.ast.SimpleQuantifier;
import org.sonarsource.analyzer.commons.regex.finders.VerboseRegexFinder;

@Rule(key = "S6353")
public class VerboseRegexCheck extends AbstractRegexCheck {

  private static final String ISSUE_MESSAGE_PATTERN = ".+syntax '(.+)' instead of.+";
  private static final Pattern issueMessagePattern = Pattern.compile(ISSUE_MESSAGE_PATTERN);
  public static final String QUICK_FIX_FORMAT = "Replace with \"%s\"";
  public static final String REDUNDANT_RANGE_MESSAGE = "Use simple character '%s' instead of '%s'.";
  public static final String REDUNDANT_REPETITION_MESSAGE = "Use simple repetition '%s' instead of '%s'.";
  public static final String REDUNDANT_REPETITION_SECONDARY_LOCATION_MESSAGE = "The repeated element.";

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    new VerboseRegexFinder(this::addIssueWithQuickFix).visit(regexParseResult);
    new PythonVerboseRegexRangeCheckVisitor().visit(regexParseResult);
    new PythonVerboseRegexRepetitionCheckVisitor().visit(regexParseResult);
  }

  public PreciseIssue addIssueWithQuickFix(RegexSyntaxElement regexTree, String message, @Nullable Integer cost, List<RegexIssueLocation> secondaries) {
    return Optional.ofNullable(addIssue(regexTree, message, cost, secondaries))
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

  private class PythonVerboseRegexRangeCheckVisitor extends RegexBaseVisitor {
    @Override
    public void visitCharacterRange(CharacterRangeTree tree) {
      var lower = tree.getLowerBound().getText();
      var upper = tree.getUpperBound().getText();
      if (upper.equals(lower)) {
        var quickFixReplacement = lower;
        var issueLocation = PythonRegexIssueLocation.preciseLocation(tree, null);
        var textEdit = new PythonTextEdit(quickFixReplacement,
          issueLocation.startLine(),
          issueLocation.startLineOffset(),
          issueLocation.endLine(),
          issueLocation.endLineOffset());

        var issue = addIssue(tree, String.format(REDUNDANT_RANGE_MESSAGE, quickFixReplacement, tree.getText()), null, Collections.emptyList());
        issue.addQuickFix(PythonQuickFix.newQuickFix(String.format(QUICK_FIX_FORMAT, quickFixReplacement), textEdit));
      }
      super.visitCharacterRange(tree);
    }
  }

  private class PythonVerboseRegexRepetitionCheckVisitor extends RegexBaseVisitor {
    @Override
    public void visit(RegexTree tree) {
      tree.continuation().toRegexTree()
        .filter(nextTree -> nextTree.is(RegexTree.Kind.REPETITION))
        .filter(RepetitionTree.class::isInstance)
        .map(RepetitionTree.class::cast)
        .filter(repetition -> repetition.getQuantifier() instanceof SimpleQuantifier)
        .filter(repetition -> ((SimpleQuantifier) repetition.getQuantifier()).getKind() == SimpleQuantifier.Kind.STAR)
        .filter(repetition -> repetition.getQuantifier().getModifier() == Quantifier.Modifier.GREEDY)
        .filter(repetition -> repetition.getRange().getBeginningOffset() > tree.getRange().getBeginningOffset())
        .ifPresent(repetition -> {
          var treeText = tree.getText();
          var nextTreeText = repetition.getElement().getText();
          if (treeText.equals(nextTreeText)) {
            var repetitionLocation = PythonRegexIssueLocation.preciseLocation(repetition, null);
            var quickFixReplacement = "+";
            var textEdit = new PythonTextEdit(quickFixReplacement,
              repetitionLocation.startLine(),
              repetitionLocation.startLineOffset(),
              repetitionLocation.endLine(),
              repetitionLocation.endLineOffset());

            var issueMessage = String.format(REDUNDANT_REPETITION_MESSAGE, treeText + quickFixReplacement, treeText + repetition.getText());
            var issue = addIssue(repetition, issueMessage, null,
              List.of(new RegexIssueLocation(tree, REDUNDANT_REPETITION_SECONDARY_LOCATION_MESSAGE)));
            issue.addQuickFix(PythonQuickFix.newQuickFix(String.format(QUICK_FIX_FORMAT, quickFixReplacement), textEdit));
          }
        });
      super.visit(tree);
    }
  }
}
