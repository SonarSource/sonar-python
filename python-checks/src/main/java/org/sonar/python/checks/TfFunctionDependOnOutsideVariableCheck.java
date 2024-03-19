/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6911")
public class TfFunctionDependOnOutsideVariableCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "This function should not depend implicitly on a global or free variable.";
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
    NameCollector collector = new NameCollector();
    functionDef.body().accept(collector);
    Set<Symbol> b = new HashSet<>(collector.symbolToNames.keySet());
    b.removeAll(functionDef.localVariables());
    if (b.isEmpty()) {
      return;
    }
    var issue = context.addIssue(functionDef.name(), MESSAGE);
    b.forEach(symbol -> collector.symbolToNames.get(symbol).forEach(name -> issue.secondary(name, MESSAGE_SECONDARY)));
  }

  private static class NameCollector extends BaseTreeVisitor {
    Map<Symbol, List<Name>> symbolToNames = new HashMap<>();

    @Override
    public void visitName(Name pyNameTree) {
      if (pyNameTree.isVariable()) {
        Optional.ofNullable(pyNameTree.symbol())
          .filter(symbol -> !symbol.is(Symbol.Kind.FUNCTION, Symbol.Kind.CLASS))
          .filter(symbol -> symbol.usages().stream().map(Usage::kind).noneMatch(usageKind -> usageKind == Usage.Kind.IMPORT))
          .ifPresent(symbol -> {
            symbolToNames.putIfAbsent(symbol, new ArrayList<>());
            symbolToNames.get(symbol).add(pyNameTree);
          });
      }
    }
  }
}
