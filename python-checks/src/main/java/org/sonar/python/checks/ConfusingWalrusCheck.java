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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FormattedExpression;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5685")
public class ConfusingWalrusCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use an assignment statement (\"=\") instead; \":=\" operator is confusing in this context.";
  private static final String MOVE_MESSAGE = "Move this assignment out of the %s; \":=\" operator is confusing in this context.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_EXPRESSION, ConfusingWalrusCheck::checkAssignmentExpression);

    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_ELEMENT, ctx -> {
      StringElement stringElement = (StringElement) ctx.syntaxNode();
      for (FormattedExpression formattedExpression : stringElement.formattedExpressions()) {
        checkNestedWalrus(ctx, formattedExpression.expression(), String.format(MOVE_MESSAGE, "interpolated expression"));
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER, ctx -> {
      Parameter parameter = (Parameter) ctx.syntaxNode();
      checkNestedWalrus(ctx, parameter, String.format(MOVE_MESSAGE, "function definition"));
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.ARG_LIST, ctx -> {
      ArgList argList = (ArgList) ctx.syntaxNode();
      if (hasKeywordArguments(argList)) {
        checkNestedWalrus(ctx, argList, String.format(MOVE_MESSAGE, "argument list"));
      }
    });
  }

  private static void checkNestedWalrus(SubscriptionContext ctx, Tree tree, String message) {
    WalrusVisitor walrusVisitor = new WalrusVisitor();
    tree.accept(walrusVisitor);
    if (!walrusVisitor.assignmentExpressions.isEmpty()) {
      PreciseIssue issue = ctx.addIssue(walrusVisitor.assignmentExpressions.get(0), message);
      walrusVisitor.assignmentExpressions.stream().skip(1).forEach(a -> issue.secondary(a, null));
    }
  }

  private static void checkAssignmentExpression(SubscriptionContext ctx) {
    AssignmentExpression assignmentExpression = (AssignmentExpression) ctx.syntaxNode();
    Optional<Tree> parentTree = Optional.ofNullable(TreeUtils.firstAncestor(assignmentExpression, a -> !a.is(Tree.Kind.PARENTHESIZED)));
    parentTree.ifPresent(parent -> {
      if (parent.is(Tree.Kind.ASSIGNMENT_STMT)) {
        ctx.addIssue(assignmentExpression, MESSAGE);
      }
      if (parent.is(Tree.Kind.EXPRESSION_STMT)) {
        ctx.addIssue(assignmentExpression, MESSAGE);
      }
    });
  }

  private static boolean hasKeywordArguments(ArgList argList) {
    for (Argument argument : argList.arguments()) {
      if (argument.is(Tree.Kind.REGULAR_ARGUMENT) && ((RegularArgument) argument).keywordArgument() != null) {
        return true;
      }
    }
    return false;
  }

  private static class WalrusVisitor extends BaseTreeVisitor {
    List<Tree> assignmentExpressions = new ArrayList<>();

    @Override
    public void visitAssignmentExpression(AssignmentExpression assignmentExpression) {
      assignmentExpressions.add(assignmentExpression);
    }
  }
}
