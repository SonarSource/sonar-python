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
      RegularArgument dataArg = TreeUtils.nthArgumentOrKeyword(0, "data", callExpression.arguments());
      if (dataArg != null && !couldExpressionBeADict(dataArg.expression())) {
        ctx.addIssue(dataArg, MESSAGE);
      }
    }
  }

  private static boolean couldExpressionBeADict(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return couldDictBeAssignedToDataArg((Name) expression, 0);
    }
    return couldTypeBeADict(expression);
  }

  private static boolean couldDictBeAssignedToDataArg(Name dataArg, int recursiveCount) {
    Symbol dataArgSymbol = dataArg.symbol();
    if (recursiveCount <= MAX_RECURSION && dataArgSymbol != null) {
      List<Tree> assignmentStmts = dataArgSymbol.usages().stream()
        .filter(usage -> usage.kind() == Usage.Kind.ASSIGNMENT_LHS)
        .map(Usage::tree)
        .map(usage -> TreeUtils.firstAncestorOfKind(usage, Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ANNOTATED_ASSIGNMENT, Tree.Kind.ASSIGNMENT_EXPRESSION))
        .filter(Objects::nonNull)
        .toList();
      if (assignmentStmts.size() == 1) {
        Tree assignment = assignmentStmts.get(0);
        Expression assignedValue = getAssignedValue(assignment);
        // We do not yet support checking the types from call expressions
        if (assignedValue != null) {
          if (assignedValue.is(Tree.Kind.NAME)) {
            return couldDictBeAssignedToDataArg((Name) assignedValue, recursiveCount + 1);
          } else {
            return couldTypeBeADict(assignedValue);
          }
        }
      }
    }
    return true;
  }

  private static boolean couldTypeBeADict(Expression expression) {
    return expression.is(Tree.Kind.DICTIONARY_LITERAL) || expression.is(Tree.Kind.DICT_COMPREHENSION) || expression.type().canBeOrExtend("dict");
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
}
