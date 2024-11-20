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
import java.util.Optional;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6711")
public class NumpyRandomStateCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a \"numpy.random.Generator\" here instead of this legacy function.";

  private static final String LEGACY_MODULE_NAME = "numpy.random.RandomState";
  private static final List<String> LEGACY_FUNCTION_EXCEPTIONS = List.of(
      "numpy.random.RandomState.get_state",
      "numpy.random.RandomState.set_state",
      "numpy.random.RandomState.seed");

  private static final List<String> LEGACY_RANDOM_FUNCTIONS = List.of(
      "numpy.random.beta",
      "numpy.random.binomial",
      "numpy.random.bytes",
      "numpy.random.chisquare",
      "numpy.random.choice",
      "numpy.random.dirichlet",
      "numpy.random.exponential",
      "numpy.random.f",
      "numpy.random.gamma",
      "numpy.random.geometric",
      "numpy.random.gumbel",
      "numpy.random.hypergeometric",
      "numpy.random.laplace",
      "numpy.random.logistic",
      "numpy.random.lognormal",
      "numpy.random.logseries",
      "numpy.random.multinomial",
      "numpy.random.multivariate_normal",
      "numpy.random.negative_binomial",
      "numpy.random.noncentral_chisquare",
      "numpy.random.noncentral_f",
      "numpy.random.normal",
      "numpy.random.pareto",
      "numpy.random.permutation",
      "numpy.random.poisson",
      "numpy.random.power",
      "numpy.random.rand",
      "numpy.random.randint",
      "numpy.random.randn",
      "numpy.random.random",
      "numpy.random.random_integers",
      "numpy.random.random_sample",
      "numpy.random.ranf",
      "numpy.random.rayleigh",
      "numpy.random.sample",
      "numpy.random.shuffle",
      "numpy.random.standard_cauchy",
      "numpy.random.standard_exponential",
      "numpy.random.standard_gamma",
      "numpy.random.standard_normal",
      "numpy.random.standard_t",
      "numpy.random.triangular",
      "numpy.random.uniform",
      "numpy.random.vonmises",
      "numpy.random.wald",
      "numpy.random.weibull",
      "numpy.random.zipf");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NumpyRandomStateCheck::checkNumpyRandomState);
  }

  private static void checkNumpyRandomState(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(call.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .filter(fqn -> !LEGACY_FUNCTION_EXCEPTIONS.contains(fqn) &&
            (fqn.startsWith(LEGACY_MODULE_NAME) || LEGACY_RANDOM_FUNCTIONS.contains(fqn)))
        .ifPresent(fqn -> ctx.addIssue(call.callee(), MESSAGE));
  }

}
