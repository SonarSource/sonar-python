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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S6742")
public class PandasChainInstructionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Refactor this long chain of instructions with \"pandas.pipe\"";
  private static final int MAX_CHAIN_LENGTH = 7;

  private static final String DATAFRAME_FQN = "pandas.core.frame.DataFrame";

  private final Set<QualifiedExpression> visited = new HashSet<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> visited.clear());
    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR, this::checkChainedInstructions);
  }

  private void checkChainedInstructions(SubscriptionContext ctx) {
    QualifiedExpression expr = (QualifiedExpression) ctx.syntaxNode();
    List<QualifiedExpression> chain = visitQualifier(expr, new ArrayList<>());
    if (chain.size() >= MAX_CHAIN_LENGTH && isValidPandasCall(chain)) {
      ctx.addIssue(chain.iterator().next(), MESSAGE);
    }
  }

  private List<QualifiedExpression> visitQualifier(QualifiedExpression expr, List<QualifiedExpression> chain) {
    if (visited.contains(expr)) {
      return chain;
    }
    visited.add(expr);
    chain.add(expr);

    if (expr.qualifier().is(Tree.Kind.CALL_EXPR)) {
      return visitCalleeQualifier((CallExpression) expr.qualifier(), chain);
    } else if (expr.qualifier().is(Tree.Kind.QUALIFIED_EXPR)) {
      return visitQualifier((QualifiedExpression) expr.qualifier(), chain);
    } else if (expr.qualifier().is(Tree.Kind.SUBSCRIPTION)) {
      return ignoreSubscriptionAndGetCallExpr((SubscriptionExpression) expr.qualifier())
        .map(callExpr -> visitCalleeQualifier(callExpr, chain))
        .orElse(chain);
    }
    return chain;
  }

  private List<QualifiedExpression> visitCalleeQualifier(CallExpression call, List<QualifiedExpression> chain) {
    return Optional.of(call.callee())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(qe -> visitQualifier(qe, chain))
      .orElse(chain);
  }

  private static Optional<CallExpression> ignoreSubscriptionAndGetCallExpr(SubscriptionExpression qualifier) {
    if (qualifier.object().is(Tree.Kind.CALL_EXPR)) {
      return Optional.of((CallExpression) qualifier.object());
    } else if (qualifier.object().is(Tree.Kind.SUBSCRIPTION)) {
      return ignoreSubscriptionAndGetCallExpr((SubscriptionExpression) qualifier.object());
    }
    return Optional.empty();
  }

  private static boolean isValidPandasCall(List<QualifiedExpression> chain) {
    QualifiedExpression firstQualifiedExpression = chain.get(chain.size() - 1);

    boolean isADataFrameMethodCall = Optional.ofNullable(firstQualifiedExpression.symbol())
      .map(Symbol::fullyQualifiedName)
      .filter(fqn -> fqn.startsWith(DATAFRAME_FQN))
      .isPresent();

    boolean isAFunctionReturningADataFrame = Optional.ofNullable(firstQualifiedExpression.symbol())
      .flatMap(PandasChainInstructionCheck::isReturnTypeADataFrame)
      .orElse(false);

    boolean doesNotContainACallToPipe = chain.stream()
      .map(QualifiedExpression::symbol)
      .filter(Objects::nonNull)
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .noneMatch((DATAFRAME_FQN + ".pipe")::equals);

    boolean isADataFrame = DATAFRAME_FQN.equals(InferredTypes.fullyQualifiedTypeName(firstQualifiedExpression.qualifier().type()));

    return (isADataFrameMethodCall || isAFunctionReturningADataFrame || isADataFrame) && doesNotContainACallToPipe;
  }

  private static Optional<Boolean> isReturnTypeADataFrame(Symbol symbol) {
    return Optional.of(symbol)
      .filter(s -> s.is(Symbol.Kind.AMBIGUOUS))
      .map(AmbiguousSymbol.class::cast)
      .map(s -> s.alternatives().stream()
        .filter(a -> a.is(Symbol.Kind.FUNCTION))
        .map(FunctionSymbol.class::cast)
        .map(FunctionSymbol::annotatedReturnTypeName)
        .anyMatch(DATAFRAME_FQN::equals))
      .or(() -> Optional.of(symbol)
        .filter(s -> s.is(Symbol.Kind.FUNCTION))
        .map(FunctionSymbol.class::cast)
        .map(FunctionSymbol::annotatedReturnTypeName)
        .map(DATAFRAME_FQN::equals));
  }

}
