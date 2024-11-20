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

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5603")
public class UnusedNestedDefinitionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Remove this unused %s declaration.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (!functionDef.decorators().isEmpty() || functionDef.isMethodDefinition()) {
        return;
      }
      checkNestedDefinition(functionDef, functionDef.name(), ctx);
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      if (!classDef.decorators().isEmpty()) {
        return;
      }
      checkNestedDefinition(classDef, classDef.name(), ctx);
    });
  }

  private static void checkNestedDefinition(Tree tree, Name name, SubscriptionContext ctx) {
    Tree parent = TreeUtils.firstAncestorOfKind(tree, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF);
    if (parent == null || !parent.is(Tree.Kind.FUNCDEF)) {
      return;
    }
    Optional.ofNullable(name.symbol()).ifPresent(s -> checkSymbolUsages(s, ctx));
  }

  private static void checkSymbolUsages(Symbol symbol, SubscriptionContext ctx) {
    if (symbol.usages().size() == 1) {
      boolean isFunction = symbol.usages().get(0).kind().equals(Usage.Kind.FUNC_DECLARATION);
      ctx.addIssue(symbol.usages().get(0).tree(), String.format(MESSAGE, isFunction ? "function" : "class"));
    }
  }
}
