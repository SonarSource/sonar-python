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

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.GlobalStatement;
import org.sonar.plugins.python.api.tree.NonlocalStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

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
  private static final TypeMatcher TF_FUNCTION = TypeMatchers.withFQN("tensorflow.function");
  private static final TypeMatcher SIDE_EFFECT_FUNCTION = TypeMatchers.withFQNSatisfying(
    fqn -> SIDE_EFFECT_FUNCTIONS.contains(fqn) ? TriBool.TRUE : TriBool.FALSE);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, this::checkFunction);
  }

  public void checkFunction(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    boolean isTfFunction = functionDef.decorators().stream().anyMatch(decorator -> isTfFunctionDecorator(decorator, ctx));
    if (!isTfFunction) {
      return;
    }
    SideEffectsVisitor sideEffectsVisitor = new SideEffectsVisitor(ctx);
    functionDef.accept(sideEffectsVisitor);
    if (!sideEffectsVisitor.forbiddenConstructs.isEmpty()) {
      PreciseIssue preciseIssue = ctx.addIssue(functionDef.name(), "Make sure this Tensorflow function doesn't have Python side effects.");
      sideEffectsVisitor.forbiddenConstructs.forEach(e -> preciseIssue.secondary(e, "Statement with side effect."));
    }
  }

  private static boolean isTfFunctionDecorator(Decorator decorator, SubscriptionContext ctx) {
    return TF_FUNCTION.isTrueFor(decorator.expression(), ctx);
  }

  static class SideEffectsVisitor extends BaseTreeVisitor {

    final List<Tree> forbiddenConstructs = new ArrayList<>();
    private final SubscriptionContext context;

    SideEffectsVisitor(SubscriptionContext context) {
      this.context = context;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (SIDE_EFFECT_FUNCTION.isTrueFor(callExpression.callee(), context)) {
        forbiddenConstructs.add(callExpression.callee());
      }
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
