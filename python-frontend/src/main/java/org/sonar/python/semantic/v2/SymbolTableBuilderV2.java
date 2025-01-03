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
package org.sonar.python.semantic.v2;

import java.util.HashMap;
import java.util.Map;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;

public class SymbolTableBuilderV2 extends BaseTreeVisitor {
  private final FileInput fileInput;
  private Map<Tree, ScopeV2> scopesByRootTree;

  public SymbolTableBuilderV2(FileInput fileInput) {
    this.fileInput = fileInput;
  }

  @Override
  public void visitFileInput(FileInput fileInput) {
    scopesByRootTree = new HashMap<>();
    fileInput.accept(new WriteUsagesVisitor(scopesByRootTree));
    fileInput.accept(new ReadUsagesVisitor(scopesByRootTree));
  }

  public SymbolTable build() {
    fileInput.accept(this);
    return new SymbolTable(scopesByRootTree);
  }
}
