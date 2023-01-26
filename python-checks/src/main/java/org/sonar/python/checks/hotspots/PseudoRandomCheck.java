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
package org.sonar.python.checks.hotspots;

import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S2245")
public class PseudoRandomCheck extends PythonSubscriptionCheck {

  private static final Set<String> FUNCTION_NAMES = Set.of(
    "random",
    "getrandbits",
    "randint",
    "sample",
    "choice",
    "choices",
    "randbytes",
    "randrange",
    "shuffle");
  private static final String RANDOM_PACKAGE_PREFIX = "random.";
  private static final String RANDOM_CLASS_PREFIX = "random.Random.";
  private static final Set<String> FUNCTIONS_TO_CHECK = getFunctionsFullyQualifiedNames();
  public static final String MESSAGE = "Make sure that using this pseudorandom number generator is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol symbol = callExpression.calleeSymbol();
      Optional.ofNullable(symbol)
        .map(Symbol::fullyQualifiedName)
        .filter(FUNCTIONS_TO_CHECK::contains)
        .ifPresent(functionFqn -> {
          ctx.addIssue(callExpression, MESSAGE);
        });
    });
  }

  private static Set<String> getFunctionsFullyQualifiedNames() {
    return FUNCTION_NAMES.stream()
      .flatMap(functionName -> Stream.of(RANDOM_PACKAGE_PREFIX + functionName, RANDOM_CLASS_PREFIX + functionName))
      .collect(Collectors.toSet());
  }

}
