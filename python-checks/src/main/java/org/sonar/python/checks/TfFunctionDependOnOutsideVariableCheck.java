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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6911")
public class TfFunctionDependOnOutsideVariableCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Make sure this function does not depend on a global or free variable.";
  private static final String MESSAGE_SECONDARY = "Variable used here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, TfFunctionDependOnOutsideVariableCheck::checkFunction);
  }

  private static void checkFunction(SubscriptionContext context) {
    FunctionDef functionDef = (FunctionDef) context.syntaxNode();

    if (!TreeUtils.isFunctionWithGivenDecoratorFQN(functionDef, "tensorflow.function")) {
      return;
    }
    NameCollector collector = new NameCollector(functionDef.localVariables());
    functionDef.body().accept(collector);
    if (collector.symbolToNames.keySet().isEmpty()) {
      return;
    }
    var issue = context.addIssue(functionDef.name(), MESSAGE);
    collector.symbolToNames.keySet().forEach(symbol -> collector.symbolToNames.get(symbol).forEach(name -> issue.secondary(name, MESSAGE_SECONDARY)));
  }

  private static class NameCollector extends BaseTreeVisitor {
    Map<Symbol, List<Name>> symbolToNames = new HashMap<>();
    Set<Symbol> localSymbols;

    private NameCollector(Set<Symbol> localSymbols) {
      this.localSymbols = localSymbols;
    }

    @Override
    public void visitName(Name pyNameTree) {
      if (localSymbols.contains(pyNameTree.symbol())) {
        return;
      }
      if (!pyNameTree.isVariable()) {
        return;
      }
      if (isInstantiatedByTensorflow(pyNameTree)) {
        return;
      }

      Optional.ofNullable(pyNameTree.symbol())
        .filter(symbol -> !symbol.is(Symbol.Kind.FUNCTION, Symbol.Kind.CLASS, Symbol.Kind.AMBIGUOUS))
        .filter(symbol -> symbol.usages().stream().map(Usage::kind).noneMatch(usageKind -> usageKind == Usage.Kind.IMPORT))
        .ifPresent(symbol -> symbolToNames.computeIfAbsent(symbol, s -> new ArrayList<>()).add(pyNameTree));
    }

    private static boolean isInstantiatedByTensorflow(Name pyNameTree) {
      Symbol symbol = pyNameTree.symbol();
      return symbol != null && symbol.usages().stream().filter(u -> u.kind().equals(Usage.Kind.ASSIGNMENT_LHS))
        .map(Usage::tree)
        .map(tree -> TreeUtils.firstAncestorOfKind(tree, Tree.Kind.ASSIGNMENT_STMT))
        .filter(Objects::nonNull)
        .map(AssignmentStatement.class::cast)
        .map(AssignmentStatement::assignedValue)
        .anyMatch(NameCollector::isTensorflowValue);
    }

    private static boolean isTensorflowValue(Expression expression) {
      return Optional.of(expression)
        .map(Expressions::removeParentheses)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
        .map(callExpression -> NameCollector.isDirectCallToTensorflow(callExpression) || NameCollector.isChainedCallToTensorflow(callExpression))
        .orElse(false);
    }

    private static boolean isChainedCallToTensorflow(CallExpression callExpression) {
      return Optional.of(callExpression).map(CallExpression::callee)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
        .map(NameCollector::getLeftMostNameFromCallChain)
        .map(name -> name.name().startsWith("tensorflow."))
        .isPresent();
    }

    private static boolean isDirectCallToTensorflow(CallExpression callExpression) {
      return Optional.of(callExpression)
        .map(CallExpression::calleeSymbol)
        .map(Symbol::fullyQualifiedName)
        .filter(fqn -> fqn.startsWith("tensorflow."))
        .isPresent();
    }

    private static Name getLeftMostNameFromCallChain(QualifiedExpression qualifiedExpression) {
      QualifiedExpression current = qualifiedExpression;
      while (current != null && current.qualifier().is(Tree.Kind.CALL_EXPR, Tree.Kind.QUALIFIED_EXPR)) {
        if (current.qualifier().is(Tree.Kind.CALL_EXPR)) {
          current = ((CallExpression) current.qualifier()).callee().is(Tree.Kind.QUALIFIED_EXPR) ? (QualifiedExpression) ((CallExpression) current.qualifier()).callee() : null;
        } else {
          current = (QualifiedExpression) current.qualifier();
        }
      }
      return current != null && current.qualifier().is(Tree.Kind.NAME) ? (Name) current.qualifier() : null;
    }
  }
}
