/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;

@Rule(key = "S8514")
public class DataclassFieldDefinitionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_MISSING_ANNOTATION = "Add a type annotation to this dataclass attribute.";

  private static final TypeMatcher IS_DATACLASS = TypeMatchers.isType("dataclasses.dataclass");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, DataclassFieldDefinitionCheck::checkClassDef);
  }

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    if (!isDataclass(classDef, ctx)) {
      return;
    }

    classDef.body().statements().forEach(statement -> checkStatement(ctx, statement));
  }

  private static boolean isDataclass(ClassDef classDef, SubscriptionContext ctx) {
    return classDef.decorators().stream().anyMatch(decorator -> isDataclassDecorator(decorator, ctx));
  }

  private static boolean isDataclassDecorator(Decorator decorator, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpr) {
      return IS_DATACLASS.isTrueFor(callExpr.callee(), ctx);
    }
    return IS_DATACLASS.isTrueFor(expression, ctx);
  }

  private static void checkStatement(SubscriptionContext ctx, Tree statement) {
    if (statement instanceof AssignmentStatement assignmentStatement && !isLikelyIntentionalClassVar(assignmentStatement)
      && !isMethodAssignment(assignmentStatement.assignedValue(), ctx)) {
      ctx.addIssue(assignmentStatement, MESSAGE_MISSING_ANNOTATION);
    }
  }

  private static boolean isMethodAssignment(Expression expression, SubscriptionContext ctx) {
    Expression value = Expressions.removeParentheses(expression);
    if (isFunction(value)) {
      return true;
    }
    // Source function return types are not always inferred. Recognize local factories
    // returning a function without exempting ordinary data-producing factory calls.
    if (value instanceof CallExpression call && call.callee() instanceof Name name && name.symbolV2() != null) {
      List<UsageV2> bindings = name.symbolV2().usages().stream().filter(UsageV2::isBindingUsage).toList();
      return bindings.size() == 1
        && bindings.get(0).kind() == UsageV2.Kind.FUNC_DECLARATION
        && bindings.get(0).tree().parent() instanceof FunctionDef function
        && isMethodFactory(function, ctx);
    }
    return false;
  }

  private static boolean isMethodFactory(FunctionDef function, SubscriptionContext ctx) {
    if (function.asyncKeyword() != null || !function.decorators().isEmpty() || hasDataReturnOrYield(function.body())) {
      return false;
    }
    // Checking explicit returns alone would miss paths that implicitly return None.
    ControlFlowGraph cfg = ctx.cfg(function);
    return cfg != null && !cfg.end().predecessors().isEmpty()
      && cfg.end().predecessors().stream().allMatch(block -> {
        List<Tree> elements = block.elements();
        return !elements.isEmpty() && returnsFunction(elements.get(elements.size() - 1));
      });
  }

  private static boolean returnsFunction(Tree tree) {
    return tree instanceof ReturnStatement statement && statement.expressions().size() == 1
      && isFunction(statement.expressions().get(0));
  }

  private static boolean hasDataReturnOrYield(Tree tree) {
    if (tree.is(Tree.Kind.YIELD_EXPR, Tree.Kind.YIELD_STMT) || (tree instanceof ReturnStatement && !returnsFunction(tree))) {
      return true;
    }
    Tree nestedBody = switch (tree) {
      case FunctionDef nested -> nested.body();
      case ClassDef nested -> nested.body();
      case LambdaExpression lambda -> lambda.expression();
      default -> null;
    };
    // Traverse arguments, return expressions and defaults, but not nested bodies.
    return tree.children().stream().filter(child -> child != nestedBody).anyMatch(DataclassFieldDefinitionCheck::hasDataReturnOrYield);
  }

  private static boolean isFunction(Expression expression) {
    Expression value = Expressions.removeParentheses(expression);
    return value.is(Tree.Kind.LAMBDA) || value.typeV2() instanceof FunctionType;
  }

  private static boolean isLikelyIntentionalClassVar(AssignmentStatement assignment) {
    List<ExpressionList> lhsList = assignment.lhsExpressions();
    if (lhsList.size() != 1) {
      return false;
    }
    List<Expression> expressions = lhsList.get(0).expressions();
    if (expressions.size() != 1 || !expressions.get(0).is(Tree.Kind.NAME)) {
      return false;
    }
    String name = ((Name) expressions.get(0)).name();
    return name.startsWith("_") || isAllCaps(name);
  }

  // ALL_CAPS names follow the Python constant convention — they are intentional class-level attributes.
  private static boolean isAllCaps(String name) {
    if (name.isEmpty()) {
      return false;
    }
    boolean hasUpperCase = false;
    for (char c : name.toCharArray()) {
      if (Character.isUpperCase(c)) {
        hasUpperCase = true;
      } else if (!Character.isDigit(c) && c != '_') {
        return false;
      }
    }
    return hasUpperCase;
  }
}
