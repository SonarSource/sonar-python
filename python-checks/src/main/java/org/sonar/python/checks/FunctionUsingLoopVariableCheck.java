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
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.FunctionLike;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.SymbolUtils;
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
    if (enclosingLoop == null || !enclosingLoop.is(Tree.Kind.WHILE_STMT, Tree.Kind.FOR_STMT, Tree.Kind.GENERATOR_EXPR, Tree.Kind.LIST_COMPREHENSION,
      Tree.Kind.SET_COMPREHENSION, Tree.Kind.DICT_COMPREHENSION)) {
      return;
    }
    if (isReturnedOrCalledWithinLoop(functionLike, enclosingLoop)) {
      return;
    }
    Set<Symbol> enclosingScopeSymbols = getEnclosingScopeSymbols(enclosingLoop);
    for (Symbol symbol : enclosingScopeSymbols) {
      List<Tree> problematicUsages = new ArrayList<>();
      List<Tree> bindingUsages = new ArrayList<>();
      for (Usage usage : symbol.usages()) {
        Tree usageTree = usage.tree();
        if (isUsedInFunctionLike(usageTree, functionLike) && !usage.isBindingUsage()) {
          if (TreeUtils.firstAncestor(usageTree, t -> t.is(Tree.Kind.NONLOCAL_STMT, Tree.Kind.GLOBAL_STMT)) != null) {
            // We don't raise any issue on the variable if it's part of a nonlocal or global statement
            problematicUsages.clear();
            break;
          }
          problematicUsages.add(usageTree);
        }
        if (usage.isBindingUsage() && isWithinEnclosingLoop(usageTree, enclosingLoop)) {
          bindingUsages.add(usageTree);
        }
      }
      reportIssue(ctx, functionLike, problematicUsages, bindingUsages, symbol.name());
    }
  }

  private static void reportIssue(SubscriptionContext ctx, FunctionLike functionLike, List<Tree> problematicUsages, List<Tree> bindingUsages, String symbolName) {
    if (!problematicUsages.isEmpty() && !bindingUsages.isEmpty()) {
      PreciseIssue issue;
      if (functionLike.is(Tree.Kind.FUNCDEF)) {
        issue = ctx.addIssue(problematicUsages.get(0), String.format("Add a parameter to function \"%s\" and use variable \"%s\" as its default value;" +
          "The value of \"%s\" might change at the next loop iteration.", ((FunctionDef) functionLike).name().name(), symbolName, symbolName))
          .secondary(((FunctionDef) functionLike).name(), "Function capturing the variable");
      } else {
        issue = ctx.addIssue(problematicUsages.get(0),
          String.format("Add a parameter to the parent lambda function and use variable \"%s\" as its default value; " +
            "The value of \"%s\" might change at the next loop iteration.", symbolName, symbolName))
          .secondary(((LambdaExpression) functionLike).lambdaKeyword(), "Lambda capturing the variable");
      }
      for (Tree bindingUsage : bindingUsages) {
        issue.secondary(bindingUsage, "Assignment in the loop");
      }
    }
  }

  private static boolean isUsedInFunctionLike(Tree usageTree, FunctionLike functionLike) {
    ParameterList parameters = functionLike.parameters();
    if (parameters != null && isUsedAsDefaultValue(usageTree, parameters)) {
      return false;
    }
    if (functionLike.is(Tree.Kind.FUNCDEF)) {
      FunctionDef functionDef = (FunctionDef) functionLike;
      for (Decorator decorator : functionDef.decorators()) {
        if (TreeUtils.hasDescendant(decorator, t -> t.equals(usageTree))) {
          return false;
        }
      }
    }
    return TreeUtils.hasDescendant(functionLike, tree -> tree.equals(usageTree));
  }

  private static boolean isUsedAsDefaultValue(Tree usageTree, ParameterList parameters) {
    return parameters.nonTuple().stream().anyMatch(p ->
      p.defaultValue() != null && (usageTree.equals(p.defaultValue()) || TreeUtils.hasDescendant(p.defaultValue(), t -> t.equals(usageTree))));
  }

  private static boolean isWithinEnclosingLoop(Tree usageTree, Tree enclosingLoop) {
    return TreeUtils.hasDescendant(enclosingLoop, tree -> tree.equals(usageTree));
  }

  private static Set<Symbol> getEnclosingScopeSymbols(Tree enclosingLoop) {
    if (enclosingLoop.is(Tree.Kind.LIST_COMPREHENSION) || enclosingLoop.is(Tree.Kind.SET_COMPREHENSION) || enclosingLoop.is(Tree.Kind.GENERATOR_EXPR)) {
      return ((ComprehensionExpression) enclosingLoop).localVariables();
    }
    if (enclosingLoop.is(Tree.Kind.DICT_COMPREHENSION)) {
      return ((DictCompExpression) enclosingLoop).localVariables();
    }
    Tree enclosingScope = TreeUtils.firstAncestor(enclosingLoop, tree -> tree.is(Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.FILE_INPUT));
    if (enclosingScope == null) {
      return Collections.emptySet();
    }
    if (enclosingScope.is(Tree.Kind.FUNCDEF)) {
      return ((FunctionLike) enclosingScope).localVariables();
    }
    if (enclosingScope.is(Tree.Kind.CLASSDEF)) {
      return ((ClassDef) enclosingScope).classFields();
    }
    return ((FileInput) enclosingScope).globalVariables();
  }

  private static Tree enclosingLoop(FunctionLike functionLike) {
    return TreeUtils.firstAncestor(functionLike, t -> t.is(Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF, Tree.Kind.WHILE_STMT, Tree.Kind.FOR_STMT,
      Tree.Kind.RETURN_STMT, Tree.Kind.YIELD_STMT, Tree.Kind.GENERATOR_EXPR, Tree.Kind.COMP_FOR,
      Tree.Kind.LIST_COMPREHENSION, Tree.Kind.SET_COMPREHENSION, Tree.Kind.DICT_COMPREHENSION));
  }

  private static boolean isReturnedOrCalledWithinLoop(FunctionLike functionLike, Tree enclosingLoop) {
    Tree parentCallExpr = TreeUtils.firstAncestor(functionLike, t -> !t.is(Tree.Kind.PARENTHESIZED));
    if (parentCallExpr != null && parentCallExpr.is(Tree.Kind.CALL_EXPR)) {
      return true;
    }
    CallOrReturnVisitor callOrReturnVisitor = new CallOrReturnVisitor(functionLike, enclosingLoop);
    enclosingLoop.accept(callOrReturnVisitor);
    return callOrReturnVisitor.isReturned || callOrReturnVisitor.isCalled;
  }

  static class CallOrReturnVisitor extends BaseTreeVisitor {
    Tree enclosingLoop;
    FunctionLike functionLike;
    boolean isReturned = false;
    boolean isCalled = false;

    public CallOrReturnVisitor(FunctionLike functionLike, Tree enclosingLoop) {
      this.functionLike = functionLike;
      this.enclosingLoop = enclosingLoop;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null) {
        if (functionLike.is(Tree.Kind.FUNCDEF)) {
          isCalled |= calleeSymbol.equals(((FunctionDef) functionLike).name().symbol());
        } else {
          // lambda expression
          Name name = variableAssigned((LambdaExpression) functionLike);
          if (name != null) {
            isCalled |= calleeSymbol.equals(name.symbol());
          }
        }
      }
      super.visitCallExpression(callExpression);
    }

    @Override
    public void visitReturnStatement(ReturnStatement returnStatement) {
      isReturned |= isFunctionLikeReturned(returnStatement);
    }

    @Override
    public void visitYieldStatement(YieldStatement yieldStatement) {
      isReturned |= isFunctionLikeReturned(yieldStatement);
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

    private boolean isFunctionLikeReturned(Tree yieldOrReturnTree) {
      if (functionLike.is(Tree.Kind.FUNCDEF)) {
        Symbol functionSymbol = ((FunctionDef) functionLike).name().symbol();
        return TreeUtils.hasDescendant(yieldOrReturnTree, d -> TreeUtils.getSymbolFromTree(d).filter(symbol -> symbol.equals(functionSymbol)).isPresent());
      } else {
        // lambda expression
        return isLambdaReturned((LambdaExpression) functionLike, yieldOrReturnTree);
      }
    }

    private static boolean isLambdaReturned(LambdaExpression lambdaExpression, Tree yieldOrReturnTree) {
      Tree parentAssignment = TreeUtils.firstAncestorOfKind(lambdaExpression, Tree.Kind.ASSIGNMENT_STMT);
      if (parentAssignment != null) {
        AssignmentStatement assignmentStatement = (AssignmentStatement) parentAssignment;
        // If the lambda expression is used to construct a returned variable, we don't raise issues to avoid FPs, even if the lambda is not returned explicitly
        return SymbolUtils.assignmentsLhs(assignmentStatement).stream()
          .map(TreeUtils::getSymbolFromTree)
          .anyMatch(symbol -> TreeUtils.hasDescendant(yieldOrReturnTree, d -> TreeUtils.getSymbolFromTree(d).equals(symbol)));
      }
      return false;
    }
  }
}
