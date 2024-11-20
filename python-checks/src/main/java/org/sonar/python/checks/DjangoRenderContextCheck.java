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

import java.util.List;
import java.util.Objects;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6556")
public class DjangoRenderContextCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use an explicit context instead of passing \"locals()\" to this Django \"render\" call.";
  private static final String SECONDARY_MESSAGE = "locals() is assigned to \"%s\" here.";
  private static final String RENDER_FUNCTION = "django.shortcuts.render";
  private static final String LOCALS = "locals";
  private static final String CONTEXT_KEYWORD = "context";

  private static final int MAX_RECURSION_COUNT = 5;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, DjangoRenderContextCheck::checkForDjangoRender);
  }

  private static void checkForDjangoRender(SubscriptionContext ctx) {
    CallExpression expression = ((CallExpression) ctx.syntaxNode());
    Symbol symbol = expression.calleeSymbol();
    if (symbol != null && RENDER_FUNCTION.equals(symbol.fullyQualifiedName())) {
      checkForContextArgument(ctx, expression);
    }
  }

  private static void checkForContextArgument(SubscriptionContext ctx, CallExpression expression) {
    RegularArgument contextArg = TreeUtils.nthArgumentOrKeyword(2, CONTEXT_KEYWORD, expression.arguments());
    if (contextArg != null) {
      if (contextArg.expression().is(Tree.Kind.CALL_EXPR)) {
        CallExpression maybeLocalsCall = (CallExpression) contextArg.expression();
        if (isLocalsCall(maybeLocalsCall)) {
          ctx.addIssue(maybeLocalsCall, MESSAGE);
        }
      } else if (contextArg.expression().is(Tree.Kind.NAME)) {
        checkIfLocalsIsAssignedToContextParameter(ctx, contextArg, (Name) contextArg.expression(), 0);
      }
    }
  }

  private static void checkIfLocalsIsAssignedToContextParameter(SubscriptionContext ctx, RegularArgument contextArg, Name maybeLocalsCall, int recursionCount) {
    if (recursionCount <= MAX_RECURSION_COUNT) {
      Symbol contextSymbol = maybeLocalsCall.symbol();
      if (contextSymbol != null) {
        List<Tree> assignmentStmts = contextSymbol.usages().stream()
          .filter(usage -> usage.kind() == Usage.Kind.ASSIGNMENT_LHS)
          .map(Usage::tree)
          .map(usage -> TreeUtils.firstAncestorOfKind(usage, Tree.Kind.ASSIGNMENT_EXPRESSION, Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ANNOTATED_ASSIGNMENT))
          .filter(Objects::nonNull)
          .toList();
        if (assignmentStmts.size() == 1) {
          Tree assignment = assignmentStmts.get(0);
          Expression assignedValue = getAssignedValue(assignment);
          if (assignedValue != null) {
            checkIfLocalsIsCalledOrFindTheNextAncestor(ctx, contextArg, assignment, assignedValue, recursionCount);
          }
        }
      }
    }
  }

  @CheckForNull
  private static Expression getAssignedValue(Tree assignment) {
    if (assignment.is(Tree.Kind.ASSIGNMENT_STMT)) {
      return ((AssignmentStatement) assignment).assignedValue();
    } else if (assignment.is(Tree.Kind.ANNOTATED_ASSIGNMENT)) {
      return ((AnnotatedAssignment) assignment).assignedValue();
    } else {
      return ((AssignmentExpression) assignment).expression();
    }
  }

  private static void checkIfLocalsIsCalledOrFindTheNextAncestor(SubscriptionContext ctx, RegularArgument contextArg, Tree assignment, Expression assignedValue,
    int recursionCount) {
    if (assignedValue.is(Tree.Kind.CALL_EXPR)) {
      CallExpression localsAssignment = (CallExpression) assignedValue;
      if (isLocalsCall(localsAssignment)) {
        raiseIssueLocalsIsAssigned(ctx, contextArg, assignment, localsAssignment);
      }
    } else if (assignedValue.is(Tree.Kind.NAME)) {
      checkIfLocalsIsAssignedToContextParameter(ctx, contextArg, (Name) assignedValue, recursionCount + 1);
    }
  }

  private static void raiseIssueLocalsIsAssigned(SubscriptionContext ctx, RegularArgument contextArg, Tree assignment, CallExpression localsAssignment) {
    PreciseIssue preciseIssue = ctx.addIssue(contextArg.expression(), MESSAGE);
    if (assignment.is(Tree.Kind.ASSIGNMENT_STMT)) {
      ((AssignmentStatement) assignment).lhsExpressions().stream().flatMap(e -> e.expressions().stream())
        .filter(expression -> expression.is(Tree.Kind.NAME))
        .map(Name.class::cast)
        .forEach(namedVariable -> preciseIssue.secondary(localsAssignment, String.format(SECONDARY_MESSAGE, namedVariable.name())));
    } else if (assignment.is(Tree.Kind.ANNOTATED_ASSIGNMENT)) {
      Expression variable = ((AnnotatedAssignment) assignment).variable();
      if (variable.is(Tree.Kind.NAME)) {
        preciseIssue.secondary(localsAssignment, String.format(SECONDARY_MESSAGE, ((Name) variable).name()));
      }
    } else {
      Name variable = ((AssignmentExpression) assignment).lhsName();
      preciseIssue.secondary(localsAssignment, String.format(SECONDARY_MESSAGE, variable.name()));
    }
  }

  private static boolean isLocalsCall(CallExpression callExpression) {
    Symbol localsSymbol = callExpression.calleeSymbol();
    return localsSymbol != null && LOCALS.equals(localsSymbol.fullyQualifiedName());
  }
}
