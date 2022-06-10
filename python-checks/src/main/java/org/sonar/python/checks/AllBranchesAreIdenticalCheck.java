/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.IfStatementImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonarsource.analyzer.commons.collections.ListUtils;

@Rule(key = "S3923")
public class AllBranchesAreIdenticalCheck extends PythonSubscriptionCheck {

  private static final List<ConditionalExpression> ignoreList = new ArrayList<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> ignoreList.clear());
    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> handleIfStatement((IfStatement) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Tree.Kind.CONDITIONAL_EXPR, ctx -> handleConditionalExpression((ConditionalExpression) ctx.syntaxNode(), ctx));
  }

  private static void handleIfStatement(IfStatement ifStmt, SubscriptionContext ctx) {
    if (ifStmt.elseBranch() == null) {
      return;
    }
    StatementList body = ifStmt.body();
    for (IfStatement elifBranch : ifStmt.elifBranches()) {
      StatementList elifBody = elifBranch.body();
      if (!CheckUtils.areEquivalent(body, elifBody)) {
        return;
      }
    }
    if (!CheckUtils.areEquivalent(body, ifStmt.elseBranch().body())) {
      return;
    }
    PreciseIssue issue = ctx.addIssue(ifStmt.keyword(), "Remove this if statement or edit its code blocks so that they're not all the same.");
    issue.secondary(issueLocation(ifStmt.body()));
    ifStmt.elifBranches().forEach(e -> issue.secondary(issueLocation(e.body())));
    issue.secondary(issueLocation(ifStmt.elseBranch().body()));
    createQuickFix((IssueWithQuickFix) issue, ifStmt, ifStmt.body().statements());
  }

  private static IssueLocation issueLocation(StatementList body) {
    List<Token> tokens = TreeUtils.nonWhitespaceTokens(body);
    return IssueLocation.preciseLocation(tokens.get(0), tokens.get(tokens.size() - 1), null);
  }

  private static void handleConditionalExpression(ConditionalExpression conditionalExpression, SubscriptionContext ctx) {
    if (ignoreList.contains(conditionalExpression)) {
      return;
    }
    if (areIdentical(conditionalExpression.trueExpression(), conditionalExpression.falseExpression())) {
      PreciseIssue issue = ctx.addIssue(conditionalExpression.ifKeyword(), "This conditional expression returns the same value whether the condition is \"true\" or \"false\".");
      addSecondaryLocations(issue, conditionalExpression.trueExpression());
      addSecondaryLocations(issue, conditionalExpression.falseExpression());
      createQuickFixConditional((IssueWithQuickFix) issue, conditionalExpression);
    }
  }

  private static void addSecondaryLocations(PreciseIssue issue, Expression expression) {
    Expression unwrappedExpression = Expressions.removeParentheses(expression);
    if (unwrappedExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
      ConditionalExpression conditionalExpression = (ConditionalExpression) unwrappedExpression;
      ignoreList.add(conditionalExpression);
      addSecondaryLocations(issue, conditionalExpression.trueExpression());
      addSecondaryLocations(issue, conditionalExpression.falseExpression());
    } else {
      issue.secondary(unwrappedExpression, null);
    }
  }

  private static boolean areIdentical(Expression trueExpression, Expression falseExpression) {
    Expression unwrappedTrueExpression = unwrapIdenticalExpressions(trueExpression);
    Expression unwrappedFalseExpression = unwrapIdenticalExpressions(falseExpression);
    return CheckUtils.areEquivalent(unwrappedTrueExpression, unwrappedFalseExpression);
  }

  private static Expression unwrapIdenticalExpressions(Expression expression) {
    Expression unwrappedExpression = Expressions.removeParentheses(expression);
    if (unwrappedExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
      boolean identicalExpressions = areIdentical(((ConditionalExpression) unwrappedExpression).trueExpression(), ((ConditionalExpression) unwrappedExpression).falseExpression());
      if (identicalExpressions) {
        while (unwrappedExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
          unwrappedExpression = Expressions.removeParentheses(((ConditionalExpression) unwrappedExpression).trueExpression());
        }
      }
    }
    return unwrappedExpression;
  }

  private static void createQuickFixConditional(IssueWithQuickFix issue, Tree tree) {
    List<Tree> children = tree.children();
    Token lastTokenOfFirst = children.get(0).lastToken();
    Token lastTokenOfConditional = children.get(children.size() - 1).lastToken();
    PythonTextEdit edit = new PythonTextEdit("", lastTokenOfFirst.line(), lastTokenOfFirst.column(),
      lastTokenOfConditional.line(), lastTokenOfConditional.column());

    PythonQuickFix quickFix = PythonQuickFix.newQuickFix("Remove the if statement")
      .addTextEdit(edit)
      .build();
    issue.addQuickFix(quickFix);
  }

  private static void createQuickFix(IssueWithQuickFix issue, Tree tree, List<Statement> statements) {
    IfStatementImpl ifStatement = (IfStatementImpl) tree;

    Token firstBodyToken = statements.get(0).firstToken();
    Token keyword = ifStatement.keyword();
    Statement lastStatement = ListUtils.getLast(statements);

    int firstLine = firstBodyToken.line();
    int lastLine = lastStatement.lastToken().line();

    Optional<ElseClause> elseBranch = Optional.ofNullable(ifStatement.elseBranch());
    Optional<Integer> lineElseBranch = elseBranch
      .map(Tree::firstToken)
      .map(Token::line);

    // lastLine is one line further if there is another if block enclosed
    if (lastStatement.lastToken().column() == 0) {
      lastLine--;
    }

    PythonQuickFix.Builder quickFixBuilder = PythonQuickFix.newQuickFix("Remove the if statement");

    // Remove the if line
    quickFixBuilder.addTextEdit(new PythonTextEdit("", keyword.line(), keyword.column(), firstBodyToken.line(), firstBodyToken.column()));

    // Remove indent from the second line until the last statement
    for (int line = firstLine + 1; line <= lastLine; line++) {
      quickFixBuilder.addTextEdit(editIndentAtLine(line));
    }

    // Remove else branch
    elseBranch.ifPresent(branch -> quickFixBuilder.addTextEdit(PythonTextEdit.remove(branch)));

    // Remove the indent on the else line
    if (ifStatement.elifBranches().isEmpty()) {
      lineElseBranch.ifPresent(lineElse -> quickFixBuilder.addTextEdit(editIndentAtLine(lineElse)));
    }

    // Take care of the elif branches, the elif branch goes up to the next else or elif branch
    for (IfStatement branch : ifStatement.elifBranches()) {
      int lineElifBranch = branch.firstToken().line();
      quickFixBuilder.addTextEdit(PythonTextEdit.remove(branch))
        .addTextEdit(editIndentAtLine(lineElifBranch));
    }

    issue.addQuickFix(quickFixBuilder.build());
  }

  private static PythonTextEdit editIndentAtLine(int line) {
    return new PythonTextEdit("", line, 0, line, 4);
  }

}
