/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6714")
public class NumpyListOverGeneratorCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Pass a list to \"np.array\" instead of passing a generator.";

  private static final TypeMatcher IS_NUMPY_ARRAY = TypeMatchers.withFQN("numpy.array");
  private static final TypeMatcher IS_OBJECT_TYPE = TypeMatchers.isType("builtins.object");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NumpyListOverGeneratorCheck::checkNumpyArrayCall);
  }

  private static void checkNumpyArrayCall(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();
    if (IS_NUMPY_ARRAY.isTrueFor(call.callee(), ctx)) {
      checkGeneratorCallee(call, ctx);
    }
  }

  private static void checkGeneratorCallee(CallExpression call, SubscriptionContext ctx) {
    List<Argument> argList = call.arguments();
    if (argList.isEmpty()) {
      return;
    }

    if (Optional.of(argList.get(0))
      .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(RegularArgument.class::cast)
      .map(RegularArgument::expression)
      .filter(regArg -> (regArg.is(Tree.Kind.GENERATOR_EXPR) || isNamedGeneratorExpression(regArg, ctx)))
      .isEmpty()) {
      return;
    }

    RegularArgument dtypeArg = TreeUtils.nthArgumentOrKeyword(1, "dtype", argList);
    if (dtypeArg == null || !IS_OBJECT_TYPE.isTrueFor(dtypeArg.expression(), ctx)) {
      ctx.addIssue(call, MESSAGE);
    }
  }

  private static boolean isNamedGeneratorExpression(Expression expression, SubscriptionContext ctx) {
    return Optional.of(expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(ctx::valuesAtLocation)
      .filter(NumpyListOverGeneratorCheck::checkSetProperties)
      .isPresent();
  }

  private static boolean checkSetProperties(Set<Expression> set) {
    return !set.isEmpty() && set.stream().allMatch(NumpyListOverGeneratorCheck::isGeneratorAndParentHasType);
  }

  private static boolean isGeneratorAndParentHasType(Expression expression) {
    return expression.is(Tree.Kind.GENERATOR_EXPR) && !expression.parent().is(Tree.Kind.ANNOTATED_ASSIGNMENT);
  }
}
