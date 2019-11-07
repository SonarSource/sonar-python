/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.semantic.Usage;

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
      checkNestedDefinition(functionDef, functionDef.name().name(), Usage.Kind.FUNC_DECLARATION, ctx);
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      if (!classDef.decorators().isEmpty()) {
        return;
      }
      checkNestedDefinition(classDef, classDef.name().name(), Usage.Kind.CLASS_DECLARATION, ctx);
    });
  }

  private static void checkNestedDefinition(Tree tree, String name, Usage.Kind kind, SubscriptionContext ctx) {
    Tree parent = tree.parent();
    while (parent != null && !parent.is(Tree.Kind.CLASSDEF) && !parent.is(Tree.Kind.FUNCDEF)) {
      parent = parent.parent();
    }
    if (parent == null || !parent.is(Tree.Kind.FUNCDEF)) {
      return;
    }
    FunctionDef parentFunction = (FunctionDef) parent;
    parentFunction.localVariables().stream().filter(s -> s.name().equals(name))
      .filter(s -> s.usages().stream().anyMatch(u -> u.kind().equals(kind)))
      .findFirst().ifPresent(s -> checkSymbolUsages(s, ctx));
  }

  private static void checkSymbolUsages(Symbol symbol, SubscriptionContext ctx) {
    if (symbol.usages().size() == 1) {
      boolean isFunction = symbol.usages().get(0).kind().equals(Usage.Kind.FUNC_DECLARATION);
      ctx.addIssue(symbol.usages().get(0).tree(), String.format(MESSAGE, isFunction ? "function" : "class"));
    }
  }
}
