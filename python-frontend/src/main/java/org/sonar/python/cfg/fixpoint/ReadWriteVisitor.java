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
package org.sonar.python.cfg.fixpoint;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;

public class ReadWriteVisitor extends BaseTreeVisitor {
  private Map<Symbol, SymbolReadWrite> symbolToUsages = new HashMap<>();

  public Map<Symbol, SymbolReadWrite> symbolToUsages() {
    return symbolToUsages;
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    Optional.ofNullable(functionDef.name().symbol()).ifPresent(symbol ->
      symbolToUsages.computeIfAbsent(symbol, s -> new SymbolReadWrite()).isWrite = true);
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
    SymbolReadWrite symbolReadWrite = symbolToUsages.getOrDefault(symbol, new SymbolReadWrite());
    symbolReadWrite.addUsage(usage);
    if (!usage.isBindingUsage()) {
      symbolReadWrite.isRead = true;
    } else if (usage.kind() == Usage.Kind.COMPOUND_ASSIGNMENT_LHS) {
      symbolReadWrite.isRead = true;
      symbolReadWrite.isWrite = true;
    } else {
      symbolReadWrite.isWrite = true;
    }
    symbolToUsages.put(symbol, symbolReadWrite);
  }

  public static final class SymbolReadWrite {
    private boolean isRead = false;
    private boolean isWrite = false;
    private List<Usage> usages = new ArrayList<>();

    public boolean isWrite() {
      return isWrite;
    }

    public boolean isRead() {
      return isRead;
    }

    public void addUsage(Usage usage) {
      usages.add(usage);
    }

    // Returns the list of symbol usages within the visited element
    public List<Usage> usages() {
      return usages;
    }
  }
}
