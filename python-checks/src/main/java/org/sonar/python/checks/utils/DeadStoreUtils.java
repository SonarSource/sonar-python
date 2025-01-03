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
package org.sonar.python.checks.utils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.LiveVariablesAnalysis.LiveVariables;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1854")
public class DeadStoreUtils {

  private DeadStoreUtils() {
    // empty constructor
  }

  /**
   * Bottom-up approach, keeping track of which variables will be read by successor elements.
   */
  public static List<UnnecessaryAssignment> findUnnecessaryAssignments(CfgBlock block, LiveVariables blockLiveVariables, FunctionDef functionDef) {
    List<UnnecessaryAssignment> unnecessaryAssignments = new ArrayList<>();
    Set<Symbol> willBeRead = new HashSet<>(blockLiveVariables.getOut());
    ListIterator<Tree> elementsReverseIterator = block.elements().listIterator(block.elements().size());
    while (elementsReverseIterator.hasPrevious()) {
      Tree element = elementsReverseIterator.previous();
      blockLiveVariables.getSymbolReadWrites(element).forEach((symbol, symbolReadWrite) -> {
        if (symbolReadWrite.isWrite() && !symbolReadWrite.isRead()) {
          if (!element.is(Tree.Kind.IMPORT_NAME) && !willBeRead.contains(symbol) && functionDef.localVariables().contains(symbol)) {
            unnecessaryAssignments.add(new UnnecessaryAssignment(symbol, element));
          }
          willBeRead.remove(symbol);
        } else if (symbolReadWrite.isRead()) {
          willBeRead.add(symbol);
        }
      });
    }
    return unnecessaryAssignments;
  }

  public static boolean isParameter(Tree element) {
    return element.is(Tree.Kind.PARAMETER) || TreeUtils.firstAncestorOfKind(element, Tree.Kind.PARAMETER) != null;
  }

  public static boolean isUsedInSubFunction(Symbol symbol, FunctionDef functionDef) {
    return symbol.usages().stream()
      .anyMatch(usage -> TreeUtils.firstAncestorOfKind(usage.tree(), Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA) != functionDef);
  }

  public static class UnnecessaryAssignment {
    public final Symbol symbol;
    public final Tree element;

    private UnnecessaryAssignment(Symbol symbol, Tree element) {
      this.symbol = symbol;
      this.element = element;
    }
  }
}
