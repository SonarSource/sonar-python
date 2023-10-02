/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.HashSet;
import java.util.List;
import java.util.Optional;

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6711")
public class NumpyRandomStateCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use a \"numpy.random.Generator\" here instead of this legacy function.";

  private static final List<String> LEGACY_MODULE_NAME = List.of("numpy.random.RandomState", "numpy.random.mtrand.RandomState");

  private static final List<String> MODULE_PREFIXES = List.of("numpy.random.%s", "numpy.random.mtrand.%s");

  private static final Set<String> LEGACY_FUNCTION_EXCEPTIONS = new HashSet<>();
  private static final Set<String> LEGACY_RANDOM_FUNCTIONS = new HashSet<>();

  static {
    MODULE_PREFIXES.forEach(m -> {
      LEGACY_FUNCTION_EXCEPTIONS.add(String.format(m, "RandomState.get_state"));
      LEGACY_FUNCTION_EXCEPTIONS.add(String.format(m, "RandomState.set_state"));
      LEGACY_FUNCTION_EXCEPTIONS.add(String.format(m, "RandomState.seed"));
    });

    MODULE_PREFIXES.forEach(m -> {
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "beta"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "binomial"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "bytes"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "chisquare"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "choice"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "dirichlet"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "exponential"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "f"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "gamma"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "geometric"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "gumbel"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "hypergeometric"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "laplace"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "logistic"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "lognormal"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "logseries"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "multinomial"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "multivariate_normal"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "negative_binomial"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "noncentral_chisquare"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "noncentral_f"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "normal"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "pareto"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "permutation"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "poisson"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "power"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "rand"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "randint"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "randn"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "random"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "random_integers"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "random_sample"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "ranf"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "rayleigh"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "sample"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "shuffle"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "standard_cauchy"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "standard_exponential"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "standard_gamma"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "standard_normal"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "standard_t"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "triangular"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "uniform"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "vonmises"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "wald"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "weibull"));
      LEGACY_RANDOM_FUNCTIONS.add(String.format(m, "zipf"));
    });
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NumpyRandomStateCheck::checkNumpyRandomState);
  }

  private static void checkNumpyRandomState(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(call.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .filter(fqn -> !LEGACY_FUNCTION_EXCEPTIONS.contains(fqn) &&
            (LEGACY_MODULE_NAME.stream().anyMatch(fqn::startsWith) || LEGACY_RANDOM_FUNCTIONS.contains(fqn)))
        .ifPresent(fqn -> ctx.addIssue(call.callee(), MESSAGE));
  }

}
