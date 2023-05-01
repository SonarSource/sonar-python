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
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6560")
public class DjangoNonDictSerializationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a dictionary object here, or set the \"safe\" flag to False.";
  private static final String JSON_RESPONSE_FUNCTION_NAME = "django.http.JsonResponse";

  private static final int MAX_RECURSION = 5;

  @Override
  public void initialize(Context context) {

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol symbol = callExpression.calleeSymbol();
      if (symbol != null && JSON_RESPONSE_FUNCTION_NAME.equals(symbol.fullyQualifiedName())) {
        checkForDictSerialization(ctx, callExpression);
      }
    });
  }

  private static void checkForDictSerialization(SubscriptionContext ctx, CallExpression callExpression) {
    RegularArgument safe = TreeUtils.nthArgumentOrKeyword(2, "safe", callExpression.arguments());
    if (safe == null || (safe.expression().is(Tree.Kind.NAME) && "True".equals(((Name) safe.expression()).name()))) {
      RegularArgument dataArg = getFirstArgument(callExpression.arguments());
      if (dataArg != null
        && (!dataArg.expression().is(Tree.Kind.DICTIONARY_LITERAL))
        && (!dataArg.expression().is(Tree.Kind.CALL_EXPR)
          && !isDictAssignedToExpression(dataArg.expression()))) {
        ctx.addIssue(dataArg, MESSAGE);
      }
    }
  }

  @CheckForNull
  private static RegularArgument getFirstArgument(List<Argument> args) {
    if (!args.isEmpty()) {
      Argument argument = args.get(0);
      if (argument.is(Tree.Kind.REGULAR_ARGUMENT)) {
        return (RegularArgument) argument;
      }
    }
    return null;
  }

  private static boolean isDictAssignedToExpression(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return couldDictBeAssignedToDataArg((Name) expression, 0);
    }
    return false;
  }

  private static boolean couldDictBeAssignedToDataArg(Name dataArg, int recursiveCount) {
    Symbol dataArgSymbol = dataArg.symbol();
    if (recursiveCount <= MAX_RECURSION && dataArgSymbol != null) {
      List<Tree> assignmentStmts = dataArgSymbol.usages().stream()
        .filter(usage -> usage.kind() == Usage.Kind.ASSIGNMENT_LHS)
        .map(Usage::tree)
        .map(usage -> TreeUtils.firstAncestorOfKind(usage, Tree.Kind.ASSIGNMENT_STMT))
        .collect(Collectors.toList());
      if (assignmentStmts.size() == 1) {
        AssignmentStatement assignment = (AssignmentStatement) assignmentStmts.get(0);
        // We do not yet support checking the types from call expressions
        if (assignment.assignedValue().is(Tree.Kind.DICTIONARY_LITERAL)
          || assignment.assignedValue().is(Tree.Kind.CALL_EXPR)
          || assignment.assignedValue().is(Tree.Kind.QUALIFIED_EXPR)) {
          return true;
        } else if (assignment.assignedValue().is(Tree.Kind.NAME)) {
          return couldDictBeAssignedToDataArg((Name) assignment.assignedValue(), recursiveCount + 1);
        } else {
          return false;
        }
      }
    }
    return true;
  }
}
