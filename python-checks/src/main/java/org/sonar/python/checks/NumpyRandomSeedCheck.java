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
import java.util.Map;
import java.util.Optional;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6709")
public class NumpyRandomSeedCheck extends PythonSubscriptionCheck {

  private static final String SEED_ARG_NAME = "seed";

  private static final Map<String, String> SEED_METHODS_TO_CHECK = Map.of(
      "numpy.seed", SEED_ARG_NAME,
      "numpy.random.seed", SEED_ARG_NAME,
      "numpy.random.default_rng", SEED_ARG_NAME,
      "numpy.random.SeedSequence", "entropy",
      "numpy.random.PCG64", SEED_ARG_NAME,
      "numpy.random.PCG64DXSM", SEED_ARG_NAME,
      "numpy.random.MT19937", SEED_ARG_NAME,
      "numpy.random.SFC64", SEED_ARG_NAME,
      "numpy.random.Philox", SEED_ARG_NAME);

  private static final String MESSAGE = "Provide a seed for this random generator.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NumpyRandomSeedCheck::checkEmptySeedCall);
  }

  private static void checkEmptySeedCall(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(call.calleeSymbol()).map(Symbol::fullyQualifiedName)
        .map(SEED_METHODS_TO_CHECK::get)
        .filter(argName -> !isSeedArgumentPresent(argName, call.arguments()))
        .ifPresent(fqn -> ctx.addIssue(call, MESSAGE));
  }

  private static boolean isSeedArgumentPresent(String argName, List<Argument> args) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, argName, args))
        .map(RegularArgument::expression)
        .filter(exp -> !exp.is(Tree.Kind.NONE)).isPresent();
  }
}
