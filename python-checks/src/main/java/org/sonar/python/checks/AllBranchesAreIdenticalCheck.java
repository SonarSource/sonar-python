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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.quickfix.TextEditUtils.removeUntil;

@Rule(key = "S3923")
public class AllBranchesAreIdenticalCheck extends PythonSubscriptionCheck {

  private static final String IF_STATEMENT_MESSAGE = "Remove this if statement or edit its code blocks so that they're not all the same.";
  private static final String CONDITIONAL_MESSAGE = "This conditional expression returns the same value whether the condition is \"true\" or \"false\".";

  private static final List<ConditionalExpression> ignoreList = new ArrayList<>();
  public static final String SECONDARY_MESSAGE = "Duplicated statements.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> ignoreList.clear());
    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, ctx -> handleIfStatement((IfStatement) ctx.syntaxNode(), ctx));
    context.registerSyntaxNodeConsumer(Tree.Kind.CONDITIONAL_EXPR, ctx -> handleConditionalExpression((ConditionalExpression) ctx.syntaxNode(), ctx));
  }

  private static void handleIfStatement(IfStatement ifStmt, SubscriptionContext ctx) {
    ElseClause elseBranch = ifStmt.elseBranch();
    if (elseBranch == null) {
      return;
    }
    StatementList body = ifStmt.body();
    for (IfStatement elifBranch : ifStmt.elifBranches()) {
      StatementList elifBody = elifBranch.body();
      if (!CheckUtils.areEquivalent(body, elifBody)) {
        return;
      }
    }
    if (!CheckUtils.areEquivalent(body, elseBranch.body())) {
      return;
    }
    PreciseIssue issue = ctx.addIssue(ifStmt.keyword(), IF_STATEMENT_MESSAGE);
    issue.secondary(secondaryIssueLocation(ifStmt.body()));
    ifStmt.elifBranches().forEach(e -> issue.secondary(secondaryIssueLocation(e.body())));
    issue.secondary(secondaryIssueLocation(elseBranch.body()));
    if (!hasSideEffect(ifStmt)) {
      issue.addQuickFix(computeQuickFixForIfStatement(ifStmt, elseBranch));
    }
  }

  private static IssueLocation secondaryIssueLocation(StatementList body) {
    List<Token> tokens = TreeUtils.nonWhitespaceTokens(body);
    return IssueLocation.preciseLocation(tokens.get(0), tokens.get(tokens.size() - 1), SECONDARY_MESSAGE);
  }

  private static void handleConditionalExpression(ConditionalExpression conditionalExpression, SubscriptionContext ctx) {
    if (ignoreList.contains(conditionalExpression)) {
      return;
    }
    if (areIdentical(conditionalExpression.trueExpression(), conditionalExpression.falseExpression())) {
      PreciseIssue issue = ctx.addIssue(conditionalExpression.ifKeyword(), CONDITIONAL_MESSAGE);
      addSecondaryLocations(issue, conditionalExpression.trueExpression());
      addSecondaryLocations(issue, conditionalExpression.falseExpression());
      issue.addQuickFix(computeQuickFixForConditional(conditionalExpression));
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
      issue.secondary(unwrappedExpression, SECONDARY_MESSAGE);
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

  /**
   * Remove everything from the conditional expect the last false expression statement.
   */
  private static PythonQuickFix computeQuickFixForConditional(ConditionalExpression conditional) {
    return PythonQuickFix.newQuickFix("Remove the if statement")
      .addTextEdit(removeUntil(conditional.firstToken(), lastFalseExpression(conditional)))
      .build();
  }

  /**
   * Conditional can be nested. To compute a proper quick fix we need to know the last false expression.
   */
  private static Tree lastFalseExpression(ConditionalExpression conditional) {
    Tree falseExpression = conditional.falseExpression();
    if (falseExpression.is(Tree.Kind.CONDITIONAL_EXPR)) {
      return lastFalseExpression((ConditionalExpression) conditional.falseExpression());
    }
    return falseExpression;
  }

  private static PythonQuickFix computeQuickFixForIfStatement(IfStatement ifStatement, ElseClause elseClause) {
    PythonQuickFix.Builder builder = PythonQuickFix.newQuickFix("Remove the if statement");

    // Remove everything from if keyword to the last branch's body
    builder.addTextEdit(removeUntil(ifStatement.keyword(), elseClause.body()));

    // Shift all body statements to the left
    // Skip first shift because already done by removeUntil of the if statement
    TextEditUtils.shiftLeft(elseClause.body()).stream()
      .skip(1)
      .forEach(builder::addTextEdit);

    return builder.build();
  }

  private static boolean hasSideEffect(IfStatement ifStatement) {
    if (containsPossibleSideEffect(ifStatement.condition())) {
      return true;
    }
    return ifStatement.elifBranches().stream()
      .map(IfStatement::condition)
      .anyMatch(AllBranchesAreIdenticalCheck::containsPossibleSideEffect);
  }

  private static boolean containsPossibleSideEffect(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      return true;
    }
    if (expression instanceof BinaryExpression binaryExpression) {
      return containsPossibleSideEffect(binaryExpression.leftOperand()) || containsPossibleSideEffect(binaryExpression.rightOperand());
    }
    if (expression instanceof ParenthesizedExpression parenthesizedExpression) {
      return containsPossibleSideEffect(parenthesizedExpression.expression());
    }
    return false;
  }
}
