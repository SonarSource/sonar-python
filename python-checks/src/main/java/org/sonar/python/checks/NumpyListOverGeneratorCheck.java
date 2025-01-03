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

import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6714")
public class NumpyListOverGeneratorCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Pass a list to \"np.array\" instead of passing a generator.";
  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis((ctx.pythonFile())));
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkNumpyArrayCall);
  }

  private void checkNumpyArrayCall(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(call.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter("numpy.array"::equals)
      .ifPresent(fqn -> checkGeneratorCallee(call, ctx));
  }

  private void checkGeneratorCallee(CallExpression call, SubscriptionContext ctx) {
    List<Argument> argList = call.arguments();
    if (argList.isEmpty()) {
      return;
    }

    if (Optional.of(argList.get(0))
      .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(RegularArgument.class::cast)
      .map(RegularArgument::expression)
      .filter(regArg -> (regArg.is(Tree.Kind.GENERATOR_EXPR) || this.isNamedGeneratorExpression(regArg)))
      .isEmpty()) {
      return;
    }

    if (Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(1, "dtype", argList))
      .filter(regArg -> regArg.expression().is(Tree.Kind.NAME))
      .map(regArg -> (Name) regArg.expression())
      .map(HasSymbol::symbol)
      .map(Symbol::fullyQualifiedName)
      .filter("object"::equals)
      .isEmpty()) {
      ctx.addIssue(call, MESSAGE);
    }
  }

  private boolean isNamedGeneratorExpression(Expression expression) {
    return Optional.of(expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(name -> this.reachingDefinitionsAnalysis.valuesAtLocation(name))
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
