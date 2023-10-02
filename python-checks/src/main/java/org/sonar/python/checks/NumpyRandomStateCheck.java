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

  private static final String LEGACY_MODULE_NAME = "numpy.random.mtrand.RandomState";
  private static final List<String> LEGACY_FUNCTION_EXCEPTIONS = List.of(
      "numpy.random.mtrand.RandomState.get_state",
      "numpy.random.mtrand.RandomState.set_state",
      "numpy.random.mtrand.RandomState.seed");

  private static final List<String> LEGACY_RANDOM_FUNCTIONS = List.of(
      "numpy.random.mtrand.beta",
      "numpy.random.mtrand.binomial",
      "numpy.random.mtrand.bytes",
      "numpy.random.mtrand.chisquare",
      "numpy.random.mtrand.choice",
      "numpy.random.mtrand.dirichlet",
      "numpy.random.mtrand.exponential",
      "numpy.random.mtrand.f",
      "numpy.random.mtrand.gamma",
      "numpy.random.mtrand.geometric",
      "numpy.random.mtrand.gumbel",
      "numpy.random.mtrand.hypergeometric",
      "numpy.random.mtrand.laplace",
      "numpy.random.mtrand.logistic",
      "numpy.random.mtrand.lognormal",
      "numpy.random.mtrand.logseries",
      "numpy.random.mtrand.multinomial",
      "numpy.random.mtrand.multivariate_normal",
      "numpy.random.mtrand.negative_binomial",
      "numpy.random.mtrand.noncentral_chisquare",
      "numpy.random.mtrand.noncentral_f",
      "numpy.random.mtrand.normal",
      "numpy.random.mtrand.pareto",
      "numpy.random.mtrand.permutation",
      "numpy.random.mtrand.poisson",
      "numpy.random.mtrand.power",
      "numpy.random.mtrand.rand",
      "numpy.random.mtrand.randint",
      "numpy.random.mtrand.randn",
      "numpy.random.mtrand.random",
      "numpy.random.mtrand.random_integers",
      "numpy.random.mtrand.random_sample",
      "numpy.random.mtrand.ranf",
      "numpy.random.mtrand.rayleigh",
      "numpy.random.mtrand.sample",
      "numpy.random.mtrand.shuffle",
      "numpy.random.mtrand.standard_cauchy",
      "numpy.random.mtrand.standard_exponential",
      "numpy.random.mtrand.standard_gamma",
      "numpy.random.mtrand.standard_normal",
      "numpy.random.mtrand.standard_t",
      "numpy.random.mtrand.triangular",
      "numpy.random.mtrand.uniform",
      "numpy.random.mtrand.vonmises",
      "numpy.random.mtrand.wald",
      "numpy.random.mtrand.weibull",
      "numpy.random.mtrand.zipf");

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
