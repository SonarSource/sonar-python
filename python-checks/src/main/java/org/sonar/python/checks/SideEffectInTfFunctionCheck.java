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
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.GlobalStatement;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.NonlocalStatement;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6928")
public class SideEffectInTfFunctionCheck extends PythonSubscriptionCheck {

  private static final List<String> SIDE_EFFECT_FUNCTIONS = List.of(
    "print",
    "open",
    "list.append",
    "list.extend",
    "list.clear",
    "list.insert",
    "list.remove",
    "set.add",
    "set.remove",
    "set.clear",
    "dict.update",
    "dict.pop",
    "dict.clear",
    "typing.MutableMapping.update",
    "typing.MutableMapping.pop",
    "typing.MutableMapping.clear"
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, this::checkFunction);
  }

  public void checkFunction(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    boolean isTfFunction = functionDef.decorators().stream().anyMatch(SideEffectInTfFunctionCheck::isTfFunctionDecorator);
    if (!isTfFunction) {
      return;
    }
    SideEffectsVisitor sideEffectsVisitor = new SideEffectsVisitor();
    functionDef.accept(sideEffectsVisitor);
    if (!sideEffectsVisitor.forbiddenConstructs.isEmpty()) {
      PreciseIssue preciseIssue = ctx.addIssue(functionDef.name(), "Make sure this Tensorflow function doesn't have Python side effects.");
      sideEffectsVisitor.forbiddenConstructs.forEach(e -> preciseIssue.secondary(e, "Statement with side effect."));
    }
  }

  private static boolean isTfFunctionDecorator(Decorator decorator) {
    Expression expression = decorator.expression();
    if (expression instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      return symbol != null && "tensorflow.function".equals(symbol.fullyQualifiedName());
    }
    return false;
  }

  static class SideEffectsVisitor extends BaseTreeVisitor {

    final List<Tree> forbiddenConstructs = new ArrayList<>();

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Optional.ofNullable(callExpression.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .filter(SIDE_EFFECT_FUNCTIONS::contains)
        .ifPresent(s -> forbiddenConstructs.add(callExpression.callee()));
      super.visitCallExpression(callExpression);
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement assignmentStatement) {
      boolean lhsContainsSubscription = assignmentStatement.lhsExpressions().stream()
        .anyMatch(exprList -> exprList.expressions().stream()
          .anyMatch(e -> e.is(Tree.Kind.SUBSCRIPTION)
          )
        );
      if (lhsContainsSubscription) {
        this.forbiddenConstructs.add(assignmentStatement);
      }
      super.visitAssignmentStatement(assignmentStatement);
    }

    @Override
    public void visitGlobalStatement(GlobalStatement globalStatement) {
      this.forbiddenConstructs.add(globalStatement);
    }

    @Override
    public void visitNonlocalStatement(NonlocalStatement nonlocalStatement) {
      this.forbiddenConstructs.add(nonlocalStatement);
    }

    @Override
    public void visitDelStatement(DelStatement delStatement) {
      this.forbiddenConstructs.add(delStatement);
    }
  }
}
