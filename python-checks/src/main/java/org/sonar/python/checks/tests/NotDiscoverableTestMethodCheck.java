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
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static java.util.stream.Stream.concat;

@Rule(key = "S5899")
public class NotDiscoverableTestMethodCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Rename this method so that it starts with \"test\" or remove this unused helper.";
  // All methods of unittest https://docs.python.org/3/library/unittest.html#unittest.TestCase
  private static final List<String> UNITTEST_RUN_METHODS = List.of("setUp", "tearDown", "setUpClass", "tearDownClass", "run", "skiptTest",
    "subTest", "debug");
  private static final List<String> UNITTEST_CHECK_METHODS = List.of("assertEqual", "assertNotEqual", "assertTrue", "assertFalse",
    "assertIs", "assertIsNot", "assertIsNone", "assertIsNotNone", "assertIn", "assertNotIn", "assertIsInstance", "assertNotIsInstance",
    "assertRaises", "assertRaisesRegex", "assertWarns", "assertWarnsRegex", "assertLogs", "assertNoLogs", "assertAlmostEqual", "assertGreater",
    "assertGreaterEqual", "assertLess", "assertLessEqual", "assertRegex", "assertNotRegex", "assertCountEqual", "addTypeEqualityFunc",
    "fail", "failureException", "longMessage", "maxDiff");
  private static final List<String> UNITTEST_GATHER_INFO = List.of("countTestCases", "defaultTestResult", "id", "shortDescription", "addCleanup",
    "doCleanups", "addClassCleanup", "doClassCleanups");
  private static final List<String> UNITTEST_METHODS = concat(concat(UNITTEST_CHECK_METHODS.stream(), UNITTEST_RUN_METHODS.stream()), UNITTEST_GATHER_INFO.stream())
    .collect(Collectors.toList());

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDefinition = (ClassDef) ctx.syntaxNode();

      if (inheritsOnlyFromUnitTest(classDefinition)) {
        Map<FunctionSymbol, FunctionDef> suspiciousFunctionsAndDefinitions = new HashMap<>();
        Set<Tree> allDefinitions = new HashSet<>();
        // We only consider method definitions, and not nested functions
        for (Statement statement : classDefinition.body().statements()) {
          if (statement.is(Tree.Kind.FUNCDEF)) {
            FunctionDef functionDef = ((FunctionDef) statement);
            String functionName = functionDef.name().name();
            Symbol symbol = functionDef.name().symbol();
            // If it doesn't override existing methods and doesn't start with test, it is added to the map
            if (!overrideExistingMethod(functionName) && !functionName.startsWith("test")) {
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
    Optional<ClassSymbol> classSymbolFromDef = Optional.ofNullable(TreeUtils.getClassSymbolFromDef(classDefinition));
    return classSymbolFromDef.filter(classSymbol -> !classSymbol.superClasses().isEmpty()).isPresent() &&
      classSymbolFromDef
      .map(ClassSymbol::superClasses)
      .stream()
      .flatMap(List::stream)
      .map(Symbol::fullyQualifiedName)
      .allMatch("unittest.case.TestCase"::equals);
  }

  private static boolean overrideExistingMethod(String functionName) {
    return UNITTEST_METHODS.contains(functionName) || functionName.startsWith("_");
  }

}
