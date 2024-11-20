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
package org.sonar.python.semantic.v2;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Map;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Tree;

public class ScopeVisitor extends BaseTreeVisitor {

  private final Map<Tree, ScopeV2> scopesByRootTree;
  private final Deque<Tree> scopeRootTrees;

  public ScopeVisitor(Map<Tree, ScopeV2> scopesByRootTree) {
    this.scopesByRootTree = scopesByRootTree;
    this.scopeRootTrees = new LinkedList<>();
  }

  Tree currentScopeRootTree() {
    return scopeRootTrees.peek();
  }

  void enterScope(Tree tree) {
    scopeRootTrees.push(tree);
  }

  Tree leaveScope() {
    return scopeRootTrees.pop();
  }

  ScopeV2 currentScope() {
    return scopesByRootTree.get(currentScopeRootTree());
  }

  void createScope(Tree tree, @Nullable ScopeV2 parent) {
    scopesByRootTree.put(tree, new ScopeV2(parent, tree));
  }

  void createAndEnterScope(Tree tree, @Nullable ScopeV2 parent) {
    createScope(tree, parent);
    enterScope(tree);
  }

}
