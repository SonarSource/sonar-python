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

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.quickfix.TextEditUtils.insertAfter;

@Rule(key="S5719")
public class InstanceAndClassMethodsAtLeastOnePositionalCheck extends PythonSubscriptionCheck {

  private static final List<String> KNOWN_CLASS_METHODS = Arrays.asList("__new__", "__init_subclass__");

  private enum MethodIssueType {
    CLASS_METHOD("Add a class parameter", "cls"),
    REGULAR_METHOD("Add a \"self\" or class parameter", "cls", "self");

    private final String message;
    private final List<String> insertions;

    MethodIssueType(String message, String... insertions) {
      this.message = message;
      this.insertions = Arrays.asList(insertions);
    }
  }

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
      .map(decorator -> TreeUtils.decoratorNameFromExpression(decorator.expression()))
      .filter(Objects::nonNull).toList();

    if (decoratorNames.contains("staticmethod") || decoratorNames.contains("abstractstaticmethod")) {
      return;
    }

    String name = functionSymbol.name();
    if (KNOWN_CLASS_METHODS.contains(name) || decoratorNames.contains("classmethod")) {
      addIssue(ctx, functionDef, MethodIssueType.CLASS_METHOD);
    } else {
      addIssue(ctx, functionDef, MethodIssueType.REGULAR_METHOD);
    }
  }

  private static void addIssue(SubscriptionContext ctx, FunctionDef functionDef, MethodIssueType type) {
    PreciseIssue issue = ctx.addIssue(functionDef.defKeyword(), functionDef.rightPar(),
      type.message);
    String separator = functionDef.parameters() == null ? "" : ", ";
    for (String insertion : type.insertions) {
      PythonQuickFix quickFix = PythonQuickFix.newQuickFix(String.format("Add '%s' as the first parameter.", insertion))
        .addTextEdit(insertAfter(functionDef.leftPar(), insertion + separator))
        .build();
      issue.addQuickFix(quickFix);
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
