/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key="S5719")
public class InstanceAndClassMethodsAtLeastOnePositionalCheck extends PythonSubscriptionCheck {

  private static final List<String> KNOWN_CLASS_METHODS = Arrays.asList("__new__", "__init_subclass__");

  private static boolean isUsageInClassBody(Usage usage, ClassDef classDef) {
    // We want all usages that are not function declarations and their closes parent is the class definition
    return usage.kind() != Usage.Kind.FUNC_DECLARATION
      && classDef.equals(TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF));
  }

  private static void handleFunctionDef(SubscriptionContext ctx, ClassDef classDef, FunctionDef functionDef) {
    List<Parameter> parameters = TreeUtils.positionalParameters(functionDef);
    if (!parameters.isEmpty()) {
      return;
    }

    ClassSymbol classSymbol = TreeUtils.getClassSymbolFromDef(classDef);
    if (classSymbol == null || classSymbol.isOrExtends("zope.interface.Interface")) {
      return;
    }

    FunctionSymbol functionSymbol = TreeUtils.getFunctionSymbolFromDef(functionDef);
    if (functionSymbol == null || functionSymbol.usages().stream().anyMatch(usage -> isUsageInClassBody(usage, classDef))) {
      return;
    }

    List<String> decoratorNames = functionDef.decorators()
      .stream()
      .map(decorator ->
        TreeUtils.decoratorNameFromExpression(decorator.expression())
      ).filter(Objects::nonNull).collect(Collectors.toList());

    if (decoratorNames.contains("staticmethod")) {
      return;
    }

    String name = functionSymbol.name();
    if (KNOWN_CLASS_METHODS.contains(name) || decoratorNames.contains("classmethod")) {
      ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), "Add a class parameter");
    } else {
      ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(), "Add a \"self\" or class parameter");
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      TreeUtils.topLevelFunctionDefs(classDef).forEach(functionDef -> handleFunctionDef(ctx, classDef, functionDef));
    });
  }
}
