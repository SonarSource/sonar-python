/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.FunctionLike;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1515")
public class FunctionUsingLoopVariableCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, FunctionUsingLoopVariableCheck::checkFunctionLike);
    context.registerSyntaxNodeConsumer(Tree.Kind.LAMBDA, FunctionUsingLoopVariableCheck::checkFunctionLike);
  }

  private static void checkFunctionLike(SubscriptionContext ctx) {
    FunctionLike functionLike = (FunctionLike) ctx.syntaxNode();
    Tree enclosingLoop = enclosingLoop(functionLike);
    if(enclosingLoop == null || !enclosingLoop.is(Tree.Kind.WHILE_STMT, Tree.Kind.FOR_STMT, Tree.Kind.GENERATOR_EXPR, Tree.Kind.LIST_COMPREHENSION)) {
      return;
    }
    if (isCalledWithinLoop(functionLike, enclosingLoop)) {
      return;
    }
    if (isReturnedInLoop(enclosingLoop, functionLike)) {
      return;
    }
    Set<Symbol> enclosingScopeSymbols = getEnclosingScopeSymbols(functionLike);
    for (Symbol symbol : enclosingScopeSymbols) {
      List<Tree> problematicUsages = new ArrayList<>();
      List<Tree> bindingUsages = new ArrayList<>();
      for (Usage usage : symbol.usages()) {
        Tree usageTree = usage.tree();
        if (isUsedInFunctionLike(usageTree, functionLike) && !usage.isBindingUsage()) {
          problematicUsages.add(usageTree);
        }
        if (usage.isBindingUsage() && isWithinEnclosingLoop(usageTree, enclosingLoop)) {
          bindingUsages.add(usageTree);
        }
      }
      reportIssue(ctx, functionLike, problematicUsages, bindingUsages);
    }
  }

  private static void reportIssue(SubscriptionContext ctx, FunctionLike functionLike, List<Tree> problematicUsages, List<Tree> bindingUsages) {
    if (!problematicUsages.isEmpty() && !bindingUsages.isEmpty()) {
      PreciseIssue issue = ctx.addIssue(problematicUsages.get(0), "Pass this variable as a parameter with a default value.")
        .secondary(bindingUsages.get(bindingUsages.size() - 1), "Assignment in the loop");
      if (functionLike.is(Tree.Kind.FUNCDEF)) {
        issue.secondary(((FunctionDef) functionLike).name(), "Function definition");
      }
    }
  }

  private static boolean isReturnedInLoop(Tree enclosingLoop, FunctionLike functionLike) {
    return TreeUtils.hasDescendant(enclosingLoop, yieldOrReturnTree -> {
      if (yieldOrReturnTree.is(Tree.Kind.RETURN_STMT, Tree.Kind.YIELD_STMT)) {
        if (functionLike.is(Tree.Kind.FUNCDEF)) {
          Symbol functionSymbol = ((FunctionDef) functionLike).name().symbol();
          return functionSymbol.usages().stream().anyMatch(usage -> TreeUtils.hasDescendant(yieldOrReturnTree, d -> d.equals(usage.tree())));
        } else {
          // lambda expression
          return isLambdaReturned((LambdaExpression) functionLike, yieldOrReturnTree);
        }
      }
      return false;
    });
  }

  private static boolean isLambdaReturned(LambdaExpression lambdaExpression, Tree yieldOrReturnTree) {
    Tree parentAssignment = TreeUtils.firstAncestorOfKind(lambdaExpression, Tree.Kind.ASSIGNMENT_STMT);
    if (parentAssignment != null) {
      AssignmentStatement assignmentStatement = (AssignmentStatement) parentAssignment;
      // If the lambda expression is used to construct a returned variable, we don't raise issues to avoid FPs, even if the lambda is not returned explicitly
      if (assignmentStatement.lhsExpressions().get(0).expressions().get(0).is(Tree.Kind.NAME)) {
        Name name = (Name) assignmentStatement.lhsExpressions().get(0).expressions().get(0);
        Symbol nameSymbol = name.symbol();
        return nameSymbol != null && nameSymbol.usages().stream().anyMatch(usage -> TreeUtils.hasDescendant(yieldOrReturnTree, d -> d.equals(usage.tree())));
      }
    }
    return false;
  }

  private static boolean isUsedInFunctionLike(Tree usageTree, FunctionLike functionLike) {
    ParameterList parameters = functionLike.parameters();
    if (parameters != null && parameters.nonTuple().stream().anyMatch(p -> usageTree.equals(p.defaultValue()))) {
      return false;
    }
    return TreeUtils.hasDescendant(functionLike, tree -> tree.equals(usageTree));
  }

  private static boolean isWithinEnclosingLoop(Tree usageTree, Tree enclosingLoop) {
    return TreeUtils.hasDescendant(enclosingLoop, tree -> tree.equals(usageTree));
  }

  private static Set<Symbol> getEnclosingScopeSymbols(FunctionLike functionLike) {
    Tree enclosingScope = TreeUtils.firstAncestor(functionLike, tree -> tree.is(Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.FILE_INPUT, Tree.Kind.LIST_COMPREHENSION,
      Tree.Kind.GENERATOR_EXPR));
    if (enclosingScope == null) {
      return Collections.emptySet();
    }
    if (enclosingScope.is(Tree.Kind.FUNCDEF)) {
      return ((FunctionLike) enclosingScope).localVariables();
    } else if (enclosingScope.is(Tree.Kind.CLASSDEF)) {
      return ((ClassDef) enclosingScope).classFields();
    } else if (enclosingScope.is(Tree.Kind.LIST_COMPREHENSION) || enclosingScope.is(Tree.Kind.GENERATOR_EXPR)) {
      return ((ComprehensionExpression) enclosingScope).localVariables();
    } else {
      return ((FileInput) enclosingScope).globalVariables();
    }
  }

  private static Tree enclosingLoop(FunctionLike functionLike) {
    return TreeUtils.firstAncestor(functionLike, t -> t.is(Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.WHILE_STMT, Tree.Kind.FOR_STMT,
      Tree.Kind.RETURN_STMT, Tree.Kind.YIELD_STMT, Tree.Kind.GENERATOR_EXPR, Tree.Kind.COMP_FOR, Tree.Kind.LIST_COMPREHENSION));
  }

  private static boolean isCalledWithinLoop(FunctionLike functionLike, Tree enclosingLoop) {
    Tree parentCallExpr = TreeUtils.firstAncestor(functionLike, t -> !t.is(Tree.Kind.PARENTHESIZED));
    if (parentCallExpr != null && parentCallExpr.is(Tree.Kind.CALL_EXPR)) {
      return true;
    }
    return TreeUtils.hasDescendant(enclosingLoop, t -> {
      if (t.is(Tree.Kind.CALL_EXPR)) {
        CallExpression callExpression = (CallExpression) t;
        Symbol calleeSymbol = callExpression.calleeSymbol();
        if (calleeSymbol == null) {
          return false;
        }
        if (functionLike.is(Tree.Kind.FUNCDEF)) {
          return ((FunctionDef) functionLike).name().symbol().equals(calleeSymbol);
        } else {
          // lambda expression
          Name name = variableAssigned((LambdaExpression) functionLike);
          if (name != null) {
            return calleeSymbol.equals(name.symbol());
          }
        }
      }
      return false;
    });
  }

  private static Name variableAssigned(LambdaExpression lambdaExpression) {
    Tree parentAssignment = TreeUtils.firstAncestorOfKind(lambdaExpression, Tree.Kind.ASSIGNMENT_STMT);
    if (parentAssignment != null) {
      AssignmentStatement assignmentStatement = (AssignmentStatement) parentAssignment;
      if (assignmentStatement.lhsExpressions().get(0).expressions().get(0).is(Tree.Kind.NAME)) {
        Name name = (Name) assignmentStatement.lhsExpressions().get(0).expressions().get(0);
        Expression expression = Expressions.singleAssignedValue(name);
        if (expression != null && expression.equals(lambdaExpression)) {
          return name;
        }
      }
    }
    return null;
  }
}
