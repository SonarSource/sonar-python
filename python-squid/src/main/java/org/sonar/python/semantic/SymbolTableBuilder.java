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
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyAnnotatedAssignmentTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyCompoundAssignmentStatementTree;
import org.sonar.python.api.tree.PyComprehensionForTree;
import org.sonar.python.api.tree.PyDecoratorTree;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyFunctionLikeTree;
import org.sonar.python.api.tree.PyGlobalStatementTree;
import org.sonar.python.api.tree.PyImportFromTree;
import org.sonar.python.api.tree.PyImportNameTree;
import org.sonar.python.api.tree.PyLambdaExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyNonlocalStatementTree;
import org.sonar.python.api.tree.PyParameterListTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.tree.BaseTreeVisitor;
import org.sonar.python.tree.PyFunctionDefTreeImpl;
import org.sonar.python.tree.PyLambdaExpressionTreeImpl;

// SymbolTable based on https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
public class SymbolTableBuilder extends BaseTreeVisitor {

  private Map<Tree, Scope> scopesByRootTree;

  @Override
  public void visitFileInput(PyFileInputTree pyFileInputTree) {
    scopesByRootTree = new HashMap<>();
    pyFileInputTree.accept(new FirstPhaseVisitor());
    pyFileInputTree.accept(new SecondPhaseVisitor());
    scopesByRootTree.values().stream()
      .filter(scope -> scope.rootTree instanceof PyFunctionLikeTree)
      .forEach(scope -> {
        PyFunctionLikeTree funcDef = (PyFunctionLikeTree) scope.rootTree;
        for (TreeSymbol symbol : scope.symbols()) {
          if (funcDef.is(Kind.LAMBDA)) {
            ((PyLambdaExpressionTreeImpl) funcDef).addLocalVariableSymbol(symbol);
          } else {
            ((PyFunctionDefTreeImpl) funcDef).addLocalVariableSymbol(symbol);
          }
        }
      });
  }

  private static class ScopeVisitor extends BaseTreeVisitor {

    private Deque<Tree> scopeRootTrees = new LinkedList<>();

    Tree currentScopeRootTree() {
      return scopeRootTrees.peek();
    }

    void enterScope(Tree tree) {
      scopeRootTrees.push(tree);
    }

    void leaveScope() {
      scopeRootTrees.pop();
    }
  }

  private class FirstPhaseVisitor extends ScopeVisitor {

    @Override
    public void visitFileInput(PyFileInputTree tree) {
      createScope(tree, null);
      enterScope(tree);
      super.visitFileInput(tree);
    }

    @Override
    public void visitLambda(PyLambdaExpressionTree pyLambdaExpressionTree) {
      createScope(pyLambdaExpressionTree, currentScope());
      enterScope(pyLambdaExpressionTree);
      createParameters(pyLambdaExpressionTree.parameters());
      super.visitLambda(pyLambdaExpressionTree);
      leaveScope();
    }

    @Override
    public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
      createScope(pyFunctionDefTree, currentScope());
      enterScope(pyFunctionDefTree);
      createParameters(pyFunctionDefTree.parameters());
      super.visitFunctionDef(pyFunctionDefTree);
      leaveScope();
    }

    @Override
    public void visitClassDef(PyClassDefTree pyClassDefTree) {
      createScope(pyClassDefTree, currentScope());
      enterScope(pyClassDefTree);
      super.visitClassDef(pyClassDefTree);
      leaveScope();
    }

    @Override
    public void visitImportName(PyImportNameTree pyImportNameTree) {
      createImportedNames(pyImportNameTree.modules());
      super.visitImportName(pyImportNameTree);
    }

    @Override
    public void visitImportFrom(PyImportFromTree pyImportFromTree) {
      createImportedNames(pyImportFromTree.importedNames());
      super.visitImportFrom(pyImportFromTree);
    }

    private void createImportedNames(List<PyAliasedNameTree> importedNames) {
      importedNames.forEach(module -> {
        if (module.alias() != null) {
          addBindingUsage(module.alias());
        } else {
          addBindingUsage(module.dottedName().names().get(0));
        }
      });
    }

    @Override
    public void visitForStatement(PyForStatementTree pyForStatementTree) {
      createLoopVariables(pyForStatementTree);
      super.visitForStatement(pyForStatementTree);
    }

    @Override
    public void visitComprehensionFor(PyComprehensionForTree tree) {
      if (tree.loopExpression().is(Tree.Kind.NAME)) {
        addBindingUsage((PyNameTree) tree.loopExpression());
      }
      super.visitComprehensionFor(tree);
    }

    private void createLoopVariables(PyForStatementTree loopTree) {
      loopTree.expressions().forEach(expr -> {
        if (expr.is(Tree.Kind.NAME)) {
          addBindingUsage((PyNameTree) expr);
        }
      });
    }

    private void createParameters(@Nullable PyParameterListTree parameterList) {
      if (parameterList == null) {
        return;
      }
      parameterList.nonTuple().forEach(param -> addBindingUsage(param.name()));
      parameterList.all().stream()
        .filter(param -> param.is(Kind.TUPLE))
        .flatMap(param -> ((PyTupleTree) param).elements().stream())
        .filter(param -> param.is(Kind.NAME))
        .map(PyNameTree.class::cast)
        .forEach(this::addBindingUsage);
    }

    @Override
    public void visitAssignmentStatement(PyAssignmentStatementTree pyAssignmentStatementTree) {
      pyAssignmentStatementTree.lhsExpressions().stream()
        .flatMap(exprList -> exprList.expressions().stream())
        .flatMap(expr -> expr.is(Kind.TUPLE) ? ((PyTupleTree) expr).elements().stream() : Stream.of(expr))
        .filter(expr -> expr.is(Kind.NAME))
        .map(PyNameTree.class::cast)
        .forEach(this::addBindingUsage);
      super.visitAssignmentStatement(pyAssignmentStatementTree);
    }

    @Override
    public void visitAnnotatedAssignment(PyAnnotatedAssignmentTree pyAnnotatedAssignmentTree) {
      if (pyAnnotatedAssignmentTree.variable().is(Kind.NAME)) {
        addBindingUsage((PyNameTree) pyAnnotatedAssignmentTree.variable());
      }
      super.visitAnnotatedAssignment(pyAnnotatedAssignmentTree);
    }

    @Override
    public void visitCompoundAssignment(PyCompoundAssignmentStatementTree pyCompoundAssignmentStatementTree) {
      if (pyCompoundAssignmentStatementTree.lhsExpression().is(Kind.NAME)) {
        addBindingUsage((PyNameTree) pyCompoundAssignmentStatementTree.lhsExpression());
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

    private void addBindingUsage(PyNameTree nameTree) {
      currentScope().addBindingUsage(nameTree);
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

    void addBindingUsage(PyNameTree nameTree) {
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
    public void visitFileInput(PyFileInputTree tree) {
      enterScope(tree);
      super.visitFileInput(tree);
    }

    @Override
    public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
      enterScope(pyFunctionDefTree);
      super.visitFunctionDef(pyFunctionDefTree);
      leaveScope();
    }

    @Override
    public void visitLambda(PyLambdaExpressionTree pyLambdaExpressionTree) {
      enterScope(pyLambdaExpressionTree);
      super.visitLambda(pyLambdaExpressionTree);
      leaveScope();
    }

    @Override
    public void visitClassDef(PyClassDefTree pyClassDefTree) {
      enterScope(pyClassDefTree);
      super.visitClassDef(pyClassDefTree);
      leaveScope();
    }

    @Override
    public void visitDecorator(PyDecoratorTree pyDecoratorTree) {
      PyNameTree nameTree = pyDecoratorTree.name().names().get(0);
      addSymbolUsage(nameTree);
      super.visitDecorator(pyDecoratorTree);
    }

    @Override
    public void visitName(PyNameTree pyNameTree) {
      if (!pyNameTree.isVariable()) {
        return;
      }
      addSymbolUsage(pyNameTree);
      super.visitName(pyNameTree);
    }

    private void addSymbolUsage(PyNameTree nameTree) {
      Scope scope = scopesByRootTree.get(currentScopeRootTree());
      SymbolImpl symbol = scope.resolve(nameTree.name());
      if (symbol != null && !symbol.usages().contains(nameTree)) {
        symbol.addUsage(nameTree);
      }
    }
  }
}
