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
package org.sonar.python.semantic;

import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyAnnotatedAssignmentTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyCompoundAssignmentStatementTree;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyGlobalStatementTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNonlocalStatementTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.tree.BaseTreeVisitor;
import org.sonar.python.tree.PyFunctionDefTreeImpl;

public class SymbolTableBuilder extends BaseTreeVisitor {

  private Map<Tree, Scope> scopesByRootTree;

  @Override
  public void visitFileInput(PyFileInputTree pyFileInputTree) {
    scopesByRootTree = new HashMap<>();
    pyFileInputTree.accept(new FirstPhaseVisitor());
    pyFileInputTree.accept(new SecondPhaseVisitor());
    scopesByRootTree.values().stream()
      .filter(scope -> scope.rootTree.is(Kind.FUNCDEF))
      .forEach(scope -> {
        PyFunctionDefTree funcDef = (PyFunctionDefTree) scope.rootTree;
        for (TreeSymbol symbol : scope.symbols()) {
          ((PyFunctionDefTreeImpl) funcDef).addLocalVariableSymbol(symbol);
        }
      });
  }

  private static class ScopeVisitor extends BaseTreeVisitor {

    private Deque<Tree> scopeRootTrees = new LinkedList<>();

    @Override
    public void visitFileInput(PyFileInputTree tree) {
      enterScope(tree);
      super.visitFileInput(tree);
    }

    @Override
    public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
      enterScope(pyFunctionDefTree);
      super.visitFunctionDef(pyFunctionDefTree);
      scopeRootTrees.pop();
    }

    Tree currentScopeRootTree() {
      return scopeRootTrees.peek();
    }

    void enterScope(Tree tree) {
      scopeRootTrees.push(tree);
    }
  }

  private class FirstPhaseVisitor extends ScopeVisitor {

    @Override
    public void visitFileInput(PyFileInputTree tree) {
      createScope(tree, null);
      super.visitFileInput(tree);
    }

    @Override
    public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
      createScope(pyFunctionDefTree, currentScope());
      super.visitFunctionDef(pyFunctionDefTree);
    }

    @Override
    public void visitAssignmentStatement(PyAssignmentStatementTree pyAssignmentStatementTree) {
      pyAssignmentStatementTree.lhsExpressions().stream()
        .flatMap(exprList -> exprList.expressions().stream())
        .flatMap(expr -> expr.is(Kind.TUPLE) ? ((PyTupleTree) expr).elements().stream() : Stream.of(expr))
        .filter(expr -> expr.is(Kind.NAME))
        .map(PyNameTree.class::cast)
        .forEach(this::addUsage);
      super.visitAssignmentStatement(pyAssignmentStatementTree);
    }

    @Override
    public void visitAnnotatedAssignment(PyAnnotatedAssignmentTree pyAnnotatedAssignmentTree) {
      if (pyAnnotatedAssignmentTree.variable().is(Kind.NAME)) {
        addUsage((PyNameTree) pyAnnotatedAssignmentTree.variable());
      }
      super.visitAnnotatedAssignment(pyAnnotatedAssignmentTree);
    }

    @Override
    public void visitCompoundAssignment(PyCompoundAssignmentStatementTree pyCompoundAssignmentStatementTree) {
      if (pyCompoundAssignmentStatementTree.lhsExpression().is(Kind.NAME)) {
        addUsage((PyNameTree) pyCompoundAssignmentStatementTree.lhsExpression());
      }
      super.visitCompoundAssignment(pyCompoundAssignmentStatementTree);
    }

    @Override
    public void visitGlobalStatement(PyGlobalStatementTree pyGlobalStatementTree) {
      pyGlobalStatementTree.variables().stream()
        .map(PyNameTree::name)
        .forEach(name -> currentScope().addGlobalName(name));
      super.visitGlobalStatement(pyGlobalStatementTree);
    }

    @Override
    public void visitNonlocalStatement(PyNonlocalStatementTree pyNonlocalStatementTree) {
      pyNonlocalStatementTree.variables().stream()
        .map(PyNameTree::name)
        .forEach(name -> currentScope().addNonLocalName(name));
      super.visitNonlocalStatement(pyNonlocalStatementTree);
    }

    private void createScope(Tree tree, @Nullable Scope parent) {
      scopesByRootTree.put(tree, new Scope(parent, tree));
    }

    private void addUsage(PyNameTree nameTree) {
      currentScope().addUsage(nameTree);
    }

    private Scope currentScope() {
      return scopesByRootTree.get(currentScopeRootTree());
    }

  }

  private static class Scope {

    private final Tree rootTree;
    private final Scope parent;
    private final Map<String, TreeSymbol> symbolsByName = new HashMap<>();
    private final Set<TreeSymbol> symbols = new HashSet<>();
    private final Set<String> globalNames = new HashSet<>();
    private final Set<String> nonlocalNames = new HashSet<>();

    private Scope(@Nullable Scope parent, Tree rootTree) {
      this.parent = parent;
      this.rootTree = rootTree;
    }

    private Set<TreeSymbol> symbols() {
      return Collections.unmodifiableSet(symbols);
    }

    void addUsage(PyNameTree nameTree) {
      String symbolName = nameTree.name();
      if (!symbolsByName.containsKey(symbolName) && !globalNames.contains(symbolName) && !nonlocalNames.contains(symbolName)) {
        SymbolImpl symbol = new SymbolImpl(symbolName);
        symbols.add(symbol);
        symbolsByName.put(symbolName, symbol);
      }
      SymbolImpl symbol = resolve(symbolName);
      if (symbol != null) {
        symbol.addUsage(nameTree);
      }
    }

    @CheckForNull
    SymbolImpl resolve(String symbolName) {
      TreeSymbol symbol = symbolsByName.get(symbolName);
      if (parent == null || symbol != null) {
        return (SymbolImpl) symbol;
      }
      return parent.resolve(symbolName);
    }

    void addGlobalName(String name) {
      globalNames.add(name);
    }

    void addNonLocalName(String name) {
      nonlocalNames.add(name);
    }
  }

  private static class SymbolImpl implements TreeSymbol {

    private final String name;
    private final Set<Tree> usages = new HashSet<>();

    private SymbolImpl(String name) {
      this.name = name;
    }

    @Override
    public String name() {
      return name;
    }

    @Override
    public Set<Tree> usages() {
      return Collections.unmodifiableSet(usages);
    }

    public void addUsage(Tree tree) {
      usages.add(tree);
    }
  }

  /**
   * Read (i.e. non-binding) usages have to be visited in a second phase.
   * They can't be visited in the same phase as write (i.e. binding) usages,
   * since a read usage may appear in the syntax tree "before" it's declared (written).
   */
  private class SecondPhaseVisitor extends ScopeVisitor {

    @Override
    public void visitName(PyNameTree pyNameTree) {
      if (!pyNameTree.isVariable()) {
        return;
      }
      super.visitName(pyNameTree);
      Scope scope = scopesByRootTree.get(currentScopeRootTree());
      SymbolImpl symbol = scope.resolve(pyNameTree.name());
      if (symbol != null && !symbol.usages().contains(pyNameTree)) {
        symbol.addUsage(pyNameTree);
      }
    }
  }
}
