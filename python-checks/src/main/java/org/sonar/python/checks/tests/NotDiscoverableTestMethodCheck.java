/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
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
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tests.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5899")
public class NotDiscoverableTestMethodCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Rename this method so that it starts with \"test\" or remove this unused helper.";
  private static final String QUICK_FIX_MESSAGE = "Rename '%s' to '%s'";
  private static final String DISCOVERABLE_TEST_PREFIX = "test_";
  private final Set<String> globalFixture = new HashSet<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> lookForGlobalFixture((FunctionDef) ctx.syntaxNode()));

    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDefinition = (ClassDef) ctx.syntaxNode();

      if (inheritsOnlyFromUnitTest(classDefinition)) {
        Map<FunctionSymbol, FunctionDef> suspiciousFunctionsAndDefinitions = new HashMap<>();
        Set<Tree> allDefinitions = new HashSet<>();
        Set<String> classFixtures = getFixturesFromClass(classDefinition);
        // We only consider method definitions, and not nested functions
        for (Statement statement : classDefinition.body().statements()) {
          if (statement.is(Tree.Kind.FUNCDEF)) {
            FunctionDef functionDef = ((FunctionDef) statement);
            if (!isException(functionDef, classFixtures)) {
              Optional.ofNullable(functionDef.name().symbol())
                .filter(symbol -> symbol.is(Symbol.Kind.FUNCTION))
                .ifPresent(symbol -> suspiciousFunctionsAndDefinitions.put(((FunctionSymbol) symbol), functionDef));
            }
            allDefinitions.add(functionDef);
          }
        }

        checkSuspiciousFunctionsUsages(ctx, suspiciousFunctionsAndDefinitions, allDefinitions);
      }
    });
  }

  private void lookForGlobalFixture(FunctionDef functionDef) {
    if (functionDef.isMethodDefinition()) {
      return;
    }
    if (functionDef.decorators().stream().anyMatch(NotDiscoverableTestMethodCheck::isPytestFixture)) {
      globalFixture.add(functionDef.name().name());
    }
  }

  /** https://docs.pytest.org/en/6.2.x/fixture.html
   * Retrieve all Fixtures defined in a class.
   * A fixture is a method which has the specific decorator @pytest.fixture
   * In those cases, pytest will invoke the method fixture and inject the result in any test method
   * for which one of their parameter name match with the fixture method name.
   */
  private static Set<String> getFixturesFromClass(ClassDef classDefinition) {
    return classDefinition.body().statements().stream()
      .filter(statement -> statement.is(Tree.Kind.FUNCDEF))
      .map(FunctionDef.class::cast)
      .filter(functionDef -> functionDef.decorators().stream().anyMatch(NotDiscoverableTestMethodCheck::isPytestFixture))
      .map(functionDef -> functionDef.name().name())
      .collect(Collectors.toSet());
  }

  private boolean isException(FunctionDef functionDef, Set<String> classFixtures) {
    String functionName = functionDef.name().name();
    return overrideExistingMethod(functionName) || functionName.startsWith("test") || isHelper(functionDef, classFixtures);
  }

  private static boolean isPytestFixture(Decorator decorator) {
    String decoratorName = TreeUtils.decoratorNameFromExpression(decorator.expression());
    return  "pytest.fixture".equals(decoratorName);
  }

  // Only raises issue when the (non-test) method is not used inside the class
  private static void checkSuspiciousFunctionsUsages(SubscriptionContext ctx, Map<FunctionSymbol, FunctionDef> suspiciousFunctionsAndDefinitions, Set<Tree> allDefinitions) {
    suspiciousFunctionsAndDefinitions.forEach((s, d) -> {
      List<Usage> usages = s.usages();
      if (usages.size() == 1 || usages.stream().noneMatch(u -> TreeUtils.firstAncestor(u.tree(), allDefinitions::contains) != null)) {
        var issue = ctx.addIssue(d.name(), MESSAGE);
        createQuickFix(d).ifPresent(issue::addQuickFix);
      }
    });
  }

  private static Optional<PythonQuickFix> createQuickFix(FunctionDef functionDef) {
    String currentName = functionDef.name().name();
    String newName = discoverableName(currentName);

    if (isAlreadyUsedInEnclosingClass(functionDef, newName)) {
      return Optional.empty();
    }

    return Optional.of(PythonQuickFix.newQuickFix(String.format(QUICK_FIX_MESSAGE, currentName, newName))
      .addTextEdit(TextEditUtils.renameAllUsages(functionDef.name(), newName))
      .build());
  }

  private static String discoverableName(String currentName) {
    return DISCOVERABLE_TEST_PREFIX + currentName;
  }

  private static boolean isAlreadyUsedInEnclosingClass(FunctionDef functionDef, String newName) {
    return Optional.ofNullable(TreeUtils.getEnclosingClassDef(functionDef)).stream()
      .flatMap(classDef -> TreeUtils.topLevelFunctionDefs(classDef).stream())
      .filter(enclosingFunctionDef -> enclosingFunctionDef != functionDef)
      .map(enclosingFunctionDef -> enclosingFunctionDef.name().name())
      .anyMatch(newName::equals);
  }

  private static boolean inheritsOnlyFromUnitTest(ClassDef classDef) {
    return TreeUtils.getParentClassesFQN(classDef).stream().anyMatch(name -> name.contains("unittest") && name.contains("TestCase"))
      && Optional.ofNullable(TreeUtils.getClassSymbolFromDef(classDef)).stream()
        .anyMatch(classSym -> classSym.superClasses().size() == 1);
  }

  private static boolean overrideExistingMethod(String functionName) {
    return UnittestUtils.allMethods().contains(functionName) || functionName.startsWith("_");
  }

  private boolean isHelper(FunctionDef functionDef, Set<String> currentClassFixture) {
    return Optional.ofNullable(TreeUtils.getFunctionSymbolFromDef(functionDef)).stream()
      .anyMatch(functionSymbol -> functionSymbol.hasDecorators() || !functionSymbol.parameters().stream()
        .map(FunctionSymbol.Parameter::name)
        .filter(Objects::nonNull)
        .allMatch(name -> "self".equals(name) || globalFixture.contains(name) || currentClassFixture.contains(name)));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
