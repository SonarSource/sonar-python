/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.checks.tests;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tests.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5899")
public class NotDiscoverableTestMethodCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Rename this method so that it starts with \"test\" or remove this unused helper.";
  private static final Set<String> globalFixture = new HashSet<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, NotDiscoverableTestMethodCheck::lookForGlobalFixture);

    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDefinition = (ClassDef) ctx.syntaxNode();

      if (inheritsOnlyFromUnitTest(classDefinition)) {
        Map<FunctionSymbol, FunctionDef> suspiciousFunctionsAndDefinitions = new HashMap<>();
        Set<Tree> allDefinitions = new HashSet<>();
        // build set of fixtures
        Set<String> currentClassFixture = classDefinition.body().statements().stream()
          .filter(statement -> statement.is(Tree.Kind.FUNCDEF))
          .map(FunctionDef.class::cast)
          .filter(functionDef -> functionDef.decorators().stream().anyMatch(NotDiscoverableTestMethodCheck::isPytestFixture))
          .map(functionDef -> functionDef.name().name())
          .collect(Collectors.toSet());

        // We only consider method definitions, and not nested functions
        for (Statement statement : classDefinition.body().statements()) {
          if (statement.is(Tree.Kind.FUNCDEF)) {
            FunctionDef functionDef = ((FunctionDef) statement);
            String functionName = functionDef.name().name();
            Symbol symbol = functionDef.name().symbol();
            // If it doesn't override existing methods, doesn't start with test and is not a helper, it is added to the map
            if (!overrideExistingMethod(functionName) && !functionName.startsWith("test") && !isHelper(functionDef, currentClassFixture)) {
              Optional.ofNullable(symbol)
                .filter(s -> s.is(Symbol.Kind.FUNCTION))
                .ifPresent(s -> suspiciousFunctionsAndDefinitions.put(((FunctionSymbol) s), functionDef));
            }
            allDefinitions.add(functionDef);
          }
        }

        checkSuspiciousFunctionsUsages(ctx, suspiciousFunctionsAndDefinitions, allDefinitions);
      }
    });
  }

  private static void lookForGlobalFixture(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (TreeUtils.firstAncestorOfKind(functionDef, Tree.Kind.CLASSDEF) != null) {
      return;
    }
    if (functionDef.decorators().stream().anyMatch(NotDiscoverableTestMethodCheck::isPytestFixture)) {
      globalFixture.add(functionDef.name().name());
    }
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static boolean isPytestFixture(Decorator decorator) {
    return Optional.of(decorator).stream()
      .map(Decorator::expression)
      .filter(expr -> expr.is(Tree.Kind.QUALIFIED_EXPR))
      .map(QualifiedExpression.class::cast)
      .filter(qualifiedExpression -> qualifiedExpression.name().name().equals("fixture"))
      .filter(qualifiedExpression -> qualifiedExpression.qualifier().is(Tree.Kind.NAME))
      .map(qualifiedExpression -> (Name) qualifiedExpression.qualifier())
      .anyMatch(name -> name.name().equals("pytest"));
  }

  // Only raises issue when the (non-test) method is not used inside the class
  private static void checkSuspiciousFunctionsUsages(SubscriptionContext ctx, Map<FunctionSymbol, FunctionDef> suspiciousFunctionsAndDefinitions, Set<Tree> allDefinitions) {
    suspiciousFunctionsAndDefinitions.forEach((s, d) -> {
      List<Usage> usages = s.usages();
      if (usages.size() == 1 || usages.stream().noneMatch(u -> TreeUtils.firstAncestor(u.tree(), allDefinitions::contains) != null)) {
        ctx.addIssue(d.name(), MESSAGE);
      }
    });
  }

  private static boolean inheritsOnlyFromUnitTest(ClassDef classDefinition) {
    return UnittestUtils.isInheritingFromUnittest(classDefinition) &&
      Optional.ofNullable(TreeUtils.getClassSymbolFromDef(classDefinition)).stream()
        .anyMatch(classSym -> classSym.superClasses().size() == 1);
  }

  private static boolean overrideExistingMethod(String functionName) {
    return UnittestUtils.allMethods().contains(functionName) || functionName.startsWith("_");
  }

  private static boolean isHelper(FunctionDef functionDef, Set<String> currentClassFixture) {
    return Optional.ofNullable(TreeUtils.getFunctionSymbolFromDef(functionDef)).stream()
      .anyMatch(functionSymbol -> functionSymbol.hasDecorators() || !functionSymbol.parameters().stream()
        .map(FunctionSymbol.Parameter::name)
        .filter(Objects::nonNull)
        .allMatch(name -> name.equals("self") || globalFixture.contains(name) || currentClassFixture.contains(name)));
  }

}
