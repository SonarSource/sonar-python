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
package org.sonar.python.checks.hotspots;

import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
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
  private static final Set<String> QUALIFIERS_TO_SKIP = Set.of("random.SystemRandom");
  private static final Set<String> FUNCTIONS_TO_CHECK = getFunctionsFullyQualifiedNames();
  public static final String MESSAGE = "Make sure that using this pseudorandom number generator is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      if (skip(callExpression)) {
        return;
      }

      Symbol symbol = callExpression.calleeSymbol();
      Optional.ofNullable(symbol)
        .map(Symbol::fullyQualifiedName)
        .filter(FUNCTIONS_TO_CHECK::contains)
        .ifPresent(functionFqn -> ctx.addIssue(callExpression, MESSAGE));
    });
  }

  private static Set<String> getFunctionsFullyQualifiedNames() {
    return FUNCTION_NAMES.stream()
      .flatMap(functionName -> Stream.of(RANDOM_PACKAGE_PREFIX + functionName, RANDOM_CLASS_PREFIX + functionName))
      .collect(Collectors.toSet());
  }

  private static boolean skip(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(CallExpression::callee)
      .filter(QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .map(QualifiedExpression::qualifier)
      .map(Expression::type)
      .filter(type -> QUALIFIERS_TO_SKIP.stream().anyMatch(type::mustBeOrExtend))
      .isPresent();
  }

}
