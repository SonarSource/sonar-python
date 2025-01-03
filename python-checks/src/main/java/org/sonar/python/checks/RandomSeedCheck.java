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

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Symbol.Kind;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6709")
public class RandomSeedCheck extends PythonSubscriptionCheck {

  private static final String NUMPY_SEED_ARG_NAME = "seed";

  private static final Map<String, String> SEED_METHODS_TO_CHECK = Map.of(
    "numpy.seed", NUMPY_SEED_ARG_NAME,
    "numpy.random.seed", NUMPY_SEED_ARG_NAME,
    "numpy.random.default_rng", NUMPY_SEED_ARG_NAME,
    "numpy.random.SeedSequence", "entropy",
    "numpy.random.PCG64", NUMPY_SEED_ARG_NAME,
    "numpy.random.PCG64DXSM", NUMPY_SEED_ARG_NAME,
    "numpy.random.MT19937", NUMPY_SEED_ARG_NAME,
    "numpy.random.SFC64", NUMPY_SEED_ARG_NAME,
    "numpy.random.Philox", NUMPY_SEED_ARG_NAME);

  private static final String SKLEARN_FQN = "sklearn";
  private static final String SKLEARN_ARG_NAME = "random_state";

  private static final String MESSAGE = "Provide a seed for this random generator.";
  private static final String SKLEARN_MESSAGE = "Provide a seed for the random_state parameter.";

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT,
      ctx -> this.reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(ctx.pythonFile()));
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkEmptySeedCall);
  }

  private void checkEmptySeedCall(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();

    Optional<Symbol> maybeCalleeSymbol = Optional.ofNullable(call.calleeSymbol());

    maybeCalleeSymbol
      .map(Symbol::fullyQualifiedName)
      .map(SEED_METHODS_TO_CHECK::get)
      .filter(argName -> isArgumentAbsentOrNone(TreeUtils.nthArgumentOrKeyword(0, argName, call.arguments())))
      .map(arg -> MESSAGE)
      .or(() -> maybeCalleeSymbol
        .filter(symbol -> symbol.fullyQualifiedName() != null && symbol.fullyQualifiedName().startsWith(SKLEARN_FQN))
        .filter(RandomSeedCheck::hasRandomStateParameter)
        .filter(symbol -> isArgumentAbsentOrNone(TreeUtils.argumentByKeyword(SKLEARN_ARG_NAME, call.arguments())))
        .map(symbol -> SKLEARN_MESSAGE))
      .ifPresent(message -> ctx.addIssue(call.callee(), message));
  }

  private static boolean hasRandomStateParameter(Symbol calleeSymbol) {
    return isClassInstantiationWithRandomStateParameter(calleeSymbol)
      .or(() -> isFunctionWithRandomStateParameter(calleeSymbol))
      .orElse(false);
  }

  private static Optional<Boolean> isClassInstantiationWithRandomStateParameter(Symbol calleeSymbol) {
    return Optional.of(calleeSymbol)
      .filter(s -> s.is(Kind.CLASS))
      .map(ClassSymbolImpl.class::cast)
      .map(classSymbol -> classSymbol.declaredMembers()
        .stream()
        .filter(member -> "__init__".equals(member.name()))
        .toList())
      .filter(members -> members.size() == 1)
      .map(members -> members.get(0))
      .map(RandomSeedCheck::hasRandomStateParameter);
  }

  private static Optional<Boolean> isFunctionWithRandomStateParameter(Symbol calleeSymbol) {
    return Optional.of(calleeSymbol)
      .filter(s1 -> s1.is(Kind.FUNCTION))
      .map(SymbolUtils::getFunctionSymbols)
      .filter(symbols -> symbols.size() == 1)
      .map(symbols -> symbols.get(0))
      .map(symbol -> symbol.parameters()
        .stream()
        .map(FunctionSymbol.Parameter::name)
        .anyMatch(SKLEARN_ARG_NAME::equals));
  }

  private boolean isArgumentAbsentOrNone(@Nullable RegularArgument arg) {
    return arg == null || arg.expression().is(Tree.Kind.NONE) || isAssignedNone(arg.expression());
  }

  private boolean isAssignedNone(Expression exp) {
    return Optional.of(exp)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(reachingDefinitionsAnalysis::valuesAtLocation)
      .filter(Predicate.not(Set::isEmpty))
      .filter(values -> values.stream().allMatch(value -> value.is(Tree.Kind.NONE))).isPresent();
  }
}
