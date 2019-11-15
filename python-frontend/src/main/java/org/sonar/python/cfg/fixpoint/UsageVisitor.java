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
package org.sonar.python.cfg.fixpoint;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;

public class UsageVisitor extends BaseTreeVisitor {
  private Map<Symbol, SymbolUsage> symbolToUsages = new HashMap<>();

  public Map<Symbol, SymbolUsage> symbolToUsages() {
    return symbolToUsages;
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    Optional.ofNullable(functionDef.name().symbol()).ifPresent(symbol ->
      symbolToUsages.computeIfAbsent(symbol, s -> new SymbolUsage()).isWrite = true);
    // don't go inside function definitions
  }

  @Override
  public void visitLambda(LambdaExpression pyLambdaExpressionTree) {
    // don't go inside lambda expressions
  }

  @Override
  public void visitName(Name name) {
    Optional.ofNullable(name.usage()).ifPresent(usage -> addToSymbolToUsageMap(usage, name.symbol()));
    super.visitName(name);
  }

  private void addToSymbolToUsageMap(Usage usage, Symbol symbol) {
    SymbolUsage symbolUsage = symbolToUsages.getOrDefault(symbol, new SymbolUsage());
    if (!usage.isBindingUsage()) {
      symbolUsage.isRead = true;
    } else if (usage.kind() == Usage.Kind.COMPOUND_ASSIGNMENT_LHS) {
      symbolUsage.isRead = true;
      symbolUsage.isWrite = true;
    } else {
      symbolUsage.isWrite = true;
    }
    symbolToUsages.put(symbol, symbolUsage);
  }

  public static final class SymbolUsage {
    private boolean isRead = false;
    private boolean isWrite = false;

    public boolean isWrite() {
      return isWrite;
    }

    public boolean isRead() {
      return isRead;
    }
  }
}
