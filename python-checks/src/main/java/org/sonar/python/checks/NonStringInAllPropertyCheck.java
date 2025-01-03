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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2823")
public class NonStringInAllPropertyCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this symbol with a string; \"__all__\" can only contain strings.";
  private static final List<String> ACCEPTED_DECORATORS = Arrays.asList("overload", "staticmethod", "classmethod");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, ctx -> {
      AssignmentStatement assignmentStatement = (AssignmentStatement) ctx.syntaxNode();
      if (TreeUtils.firstAncestorOfKind(assignmentStatement, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF) != null) {
        // We only consider __all__ assignment at module level
        return;
      }
      ExpressionList expressionList = assignmentStatement.lhsExpressions().get(0);
      if (expressionList.expressions().size() > 1) {
        return;
      }
      Expression lhs = expressionList.expressions().get(0);
      if (lhs.is(Tree.Kind.NAME) && ((Name) lhs).name().equals("__all__")) {
        checkAllProperty(ctx, assignmentStatement);
      }
    });
  }

  private static void checkAllProperty(SubscriptionContext ctx, AssignmentStatement assignmentStatement) {
    Expression assignedValue = assignmentStatement.assignedValue();
    if (!assignedValue.is(Tree.Kind.LIST_LITERAL) && !assignedValue.is(Tree.Kind.TUPLE)) {
      return;
    }
    List<Expression> expressions = getAllExpressions(assignedValue);
    for (Expression element : expressions) {
      if (!couldBeString(element, new HashSet<>())) {
        ctx.addIssue(element, MESSAGE);
      }
    }
  }

  private static boolean isClassOrFunctionSymbol(@Nullable Symbol symbol) {
    if (symbol == null) {
      return false;
    }
    if (symbol.is(Symbol.Kind.CLASS)) {
      return true;
    }
    if (symbol.is(Symbol.Kind.FUNCTION)) {
      return ACCEPTED_DECORATORS.containsAll(((FunctionSymbol) symbol).decorators());
    }
    if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
      return ((AmbiguousSymbol) symbol).alternatives().stream().allMatch(NonStringInAllPropertyCheck::isClassOrFunctionSymbol);
    }
    return false;
  }

  private static List<Expression> getAllExpressions(Expression expression) {
    if (expression.is(Tree.Kind.LIST_LITERAL)) {
      return ((ListLiteral) expression).elements().expressions();
    }
    return ((Tuple) expression).elements();
  }

  private static boolean couldBeString(Expression expression, Set<Tree> visitedTrees) {
    if (expression instanceof HasSymbol hasSymbol && isClassOrFunctionSymbol(hasSymbol.symbol())) {
      return false;
    }
    if (!expression.type().canBeOrExtend("str")) {
      return false;
    }
    if (expression.is(Tree.Kind.LAMBDA)) {
      return false;
    }
    if (expression.is(Tree.Kind.NAME) && !visitedTrees.contains(expression)) {
      visitedTrees.add(expression);
      Expression assignedValue = Expressions.singleAssignedValue((Name) expression);
      return assignedValue == null || couldBeString(assignedValue, visitedTrees);
    }
    return true;
  }
}
