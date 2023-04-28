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
package org.sonar.python.checks;

import java.util.List;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6556")
public class DjangoRenderContextCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use an explicit context instead of passing \"locals()\" to this Django \"render\" call.";
  private static final String SECONDARY_MESSAGE = "locals() is assigned to \"%s\" here";
  private static final String RENDER_FUNCTION = "django.shortcuts.render";
  private static final String LOCALS = "locals";
  private static final String CONTEXT_KEYWORD = "context";

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
        raiseIssueLocalsCallDirectly(ctx, contextArg);
      } else if (contextArg.expression().is(Tree.Kind.NAME)) {
        checkIfLocalsIsAssignedToContextParameter(ctx, contextArg);
      }
    }
  }

  private static void raiseIssueLocalsCallDirectly(SubscriptionContext ctx, RegularArgument contextArg) {
    CallExpression maybeLocalsCall = (CallExpression) contextArg.expression();
    Symbol localsSymbol = maybeLocalsCall.calleeSymbol();
    if (localsSymbol != null && LOCALS.equals(localsSymbol.fullyQualifiedName())) {
      ctx.addIssue(maybeLocalsCall, MESSAGE);
    }
  }

  private static void checkIfLocalsIsAssignedToContextParameter(SubscriptionContext ctx, RegularArgument contextArg) {
    Name maybeLocalsCall = (Name) contextArg.expression();
    Symbol contextSymbol = maybeLocalsCall.symbol();
    if (contextSymbol != null) {
      List<Tree> assignmentStmts = contextSymbol.usages().stream()
        .filter(usage -> usage.kind() == Usage.Kind.ASSIGNMENT_LHS)
        .map(Usage::tree)
        .map(usage -> TreeUtils.firstAncestorOfKind(usage, Tree.Kind.ASSIGNMENT_STMT))
        .collect(Collectors.toList());
      if (assignmentStmts.size() == 1) {
        raiseIssueLocalsIsAssigned(ctx, contextArg, assignmentStmts);
      }
    }
  }

  private static void raiseIssueLocalsIsAssigned(SubscriptionContext ctx, RegularArgument contextArg, List<Tree> assignmentStmts) {
    AssignmentStatement assignment = (AssignmentStatement) assignmentStmts.get(0);
    if (assignment.assignedValue().is(Tree.Kind.CALL_EXPR)) {
      CallExpression localsAssignment = (CallExpression) assignment.assignedValue();
      Symbol localsSymbol = localsAssignment.calleeSymbol();
      if (localsSymbol != null && LOCALS.equals(localsSymbol.fullyQualifiedName())) {
        PreciseIssue preciseIssue = ctx.addIssue(contextArg.expression(), MESSAGE);
        assignment.lhsExpressions().stream().flatMap(e -> e.expressions().stream())
          .forEach(variable -> preciseIssue.secondary(localsAssignment, String.format(SECONDARY_MESSAGE, variable)));
      }
    }
  }

}
