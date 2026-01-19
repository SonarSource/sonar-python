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
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7508")
public class NestedCollectionsCreationCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Remove this redundant call.";
  private static final String SECONDARY_MESSAGE = "A redundant call is done here.";
  private static final String QUICK_FIX_MESSAGE = MESSAGE;

  private static final String LIST_FQN = "list";
  private static final String SORTED_FQN = "sorted";
  private static final String SET_FQN = "set";
  private static final String TUPLE_FQN = "tuple";
  private static final String REVERSED_FQN = "reversed";

  private static final Map<String, Set<String>> SENSITIVE_NESTED_CALL_COMBINATIONS = Map.ofEntries(
    Map.entry(LIST_FQN, Set.of(LIST_FQN, TUPLE_FQN, SORTED_FQN)),
    Map.entry(SET_FQN, Set.of(LIST_FQN, SET_FQN, TUPLE_FQN, REVERSED_FQN, SORTED_FQN)),
    Map.entry(SORTED_FQN, Set.of(LIST_FQN, TUPLE_FQN, SORTED_FQN)),
    Map.entry(TUPLE_FQN, Set.of(LIST_FQN, TUPLE_FQN)));

  private TypeCheckMap<Set<TypeCheckBuilder>> sensitiveCallCombinationChecks;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::check);
  }

  private void initChecks(SubscriptionContext ctx) {
    sensitiveCallCombinationChecks = new TypeCheckMap<>();
    SENSITIVE_NESTED_CALL_COMBINATIONS.forEach((outerCallFqn, innerCallFqns) -> {
      var outerCallTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName(outerCallFqn);
      var innerCallTypeChecks = innerCallFqns.stream()
        .map(sensitiveMethodFqn -> ctx.typeChecker().typeCheckBuilder().isTypeWithName(sensitiveMethodFqn))
        .collect(Collectors.toSet());
      sensitiveCallCombinationChecks.put(outerCallTypeCheck, innerCallTypeChecks);
    });
  }

  private void check(SubscriptionContext ctx) {
    var callExpression = (CallExpression) ctx.syntaxNode();
    sensitiveCallCombinationChecks.getOptionalForType(callExpression.callee().typeV2())
      .ifPresent(nestedCallTypeChecks -> TreeUtils.nthArgumentOrKeywordOptional(0, "", callExpression.arguments())
        .map(RegularArgument::expression)
        .ifPresent(argumentExpression -> checkCallArgumentsForSensitiveMethods(callExpression, argumentExpression, nestedCallTypeChecks, ctx)));
  }

  private static void checkCallArgumentsForSensitiveMethods(CallExpression callExpression, Expression argumentExpression, Set<TypeCheckBuilder> nestedCallTypeChecks,
    SubscriptionContext ctx) {
    findSensitiveMethodCall(argumentExpression, nestedCallTypeChecks)
      .ifPresent(redundantCallExpression -> {
        var issue = ctx.addIssue(redundantCallExpression.callee(), MESSAGE)
          .secondary(callExpression.callee(), SECONDARY_MESSAGE);
        createQuickFix(redundantCallExpression).ifPresent(issue::addQuickFix);
      });
    findAssignedToSensitiveMethodCall(argumentExpression, nestedCallTypeChecks)
      .ifPresent(argumentAssignedCall -> {
        var issue = ctx.addIssue(argumentAssignedCall.callee(), MESSAGE)
          .secondary(callExpression.callee(), SECONDARY_MESSAGE);
        createAssignedQuickFix(argumentExpression, argumentAssignedCall).ifPresent(issue::addQuickFix);
      });
  }

  private static Optional<CallExpression> findSensitiveMethodCall(@Nullable Expression expression,
    Set<TypeCheckBuilder> sensitiveMethodsTypeChecks) {
    return TreeUtils.toOptionalInstanceOf(CallExpression.class, expression)
      .filter(callExpression -> sensitiveMethodsTypeChecks.stream()
        .map(check -> check.check(callExpression.callee().typeV2()))
        .anyMatch(TriBool.TRUE::equals));
  }

  private static Optional<CallExpression> findAssignedToSensitiveMethodCall(Expression argumentExpression, Set<TypeCheckBuilder> sensitiveMethodsTypeChecks) {
    return TreeUtils.toOptionalInstanceOf(Name.class, argumentExpression)
      .filter(name -> getUsageCount(name) == 2)
      .flatMap(name -> findSensitiveMethodCall(Expressions.singleAssignedValue(name), sensitiveMethodsTypeChecks));
  }

  private static int getUsageCount(Name name) {
    return Optional.ofNullable(name.symbolV2())
      .map(SymbolV2::usages)
      .map(List::size)
      .orElse(0);
  }

  private static Optional<PythonQuickFix> createQuickFix(Expression argumentCall) {
    return TreeUtils.toOptionalInstanceOf(CallExpression.class, argumentCall)
      .map(CallExpression::argumentList)
      .map(argList -> TreeUtils.treeToString(argList, false))
      .map(replacementText -> TextEditUtils.replace(argumentCall, replacementText))
      .map(textEdit -> PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE).addTextEdit(textEdit))
      .map(PythonQuickFix.Builder::build);
  }

  private static Optional<PythonQuickFix> createAssignedQuickFix(Expression argumentExpression, Expression argumentAssignedValueCall) {
    return TreeUtils.toOptionalInstanceOf(CallExpression.class, argumentAssignedValueCall)
      .map(CallExpression::argumentList)
      .map(argList -> TreeUtils.treeToString(argList, false))
      .map(replacementText -> {
        var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.replace(argumentExpression, replacementText));
        TreeUtils.toOptionalInstanceOf(Statement.class, TreeUtils.firstAncestor(argumentAssignedValueCall, Statement.class::isInstance))
          .map(TextEditUtils::removeStatement)
          .ifPresent(quickFix::addTextEdit);
        return quickFix.build();
      });
  }
}
