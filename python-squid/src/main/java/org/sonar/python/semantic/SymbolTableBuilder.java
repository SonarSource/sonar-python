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

import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.HasSymbol;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyAnnotatedAssignmentTree;
import org.sonar.python.api.tree.PyAnyParameterTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyCompoundAssignmentStatementTree;
import org.sonar.python.api.tree.PyComprehensionForTree;
import org.sonar.python.api.tree.PyDecoratorTree;
import org.sonar.python.api.tree.PyDottedNameTree;
import org.sonar.python.api.tree.PyExpressionTree;
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
import org.sonar.python.api.tree.PyParameterTree;
import org.sonar.python.api.tree.PyQualifiedExpressionTree;
import org.sonar.python.api.tree.PyTupleTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.tree.BaseTreeVisitor;
import org.sonar.python.tree.PyClassDefTreeImpl;
import org.sonar.python.tree.PyFunctionDefTreeImpl;
import org.sonar.python.tree.PyLambdaExpressionTreeImpl;
import org.sonar.python.tree.PyNameTreeImpl;

// SymbolTable based on https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
public class SymbolTableBuilder extends BaseTreeVisitor {

  private Map<Tree, Scope> scopesByRootTree;
  private Set<Tree> assignmentLeftHandSides = new HashSet<>();

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
    scopesByRootTree.values().stream()
      .filter(scope -> scope.rootTree.is(Kind.CLASSDEF))
      .forEach(scope -> {
        PyClassDefTreeImpl classDef = (PyClassDefTreeImpl) scope.rootTree;
        scope.symbols.forEach(classDef::addClassField);
        scope.instanceAttributesByName.values().forEach(classDef::addInstanceField);
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
      createParameters(pyLambdaExpressionTree);
      super.visitLambda(pyLambdaExpressionTree);
      leaveScope();
    }

    @Override
    public void visitFunctionDef(PyFunctionDefTree pyFunctionDefTree) {
      createScope(pyFunctionDefTree, currentScope());
      enterScope(pyFunctionDefTree);
      createParameters(pyFunctionDefTree);
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
      createImportedNames(pyImportNameTree.modules(), null, false);
      super.visitImportName(pyImportNameTree);
    }

    @Override
    public void visitImportFrom(PyImportFromTree pyImportFromTree) {
      PyDottedNameTree moduleTree = pyImportFromTree.module();
      String moduleName = moduleTree != null
        ? moduleTree.names().stream().map(PyNameTree::name).collect(Collectors.joining("."))
        : null;
      createImportedNames(pyImportFromTree.importedNames(), moduleName, !pyImportFromTree.dottedPrefixForModule().isEmpty());
      super.visitImportFrom(pyImportFromTree);
    }

    private void createImportedNames(List<PyAliasedNameTree> importedNames, @Nullable String fromModuleName, boolean isRelativeImport) {
      importedNames.forEach(module -> {
        PyNameTree nameTree = module.dottedName().names().get(0);
        String fullyQualifiedName = fromModuleName != null
          ? (fromModuleName + "." + nameTree.name())
          : nameTree.name();
        if (isRelativeImport) {
          fullyQualifiedName = null;
        }
        if (module.alias() != null) {
          addBindingUsage(module.alias(), Usage.Kind.IMPORT, fullyQualifiedName);
        } else {
          addBindingUsage(nameTree, Usage.Kind.IMPORT, fullyQualifiedName);
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
        addBindingUsage((PyNameTree) tree.loopExpression(), Usage.Kind.COMP_DECLARATION);
      }
      super.visitComprehensionFor(tree);
    }

    private void createLoopVariables(PyForStatementTree loopTree) {
      loopTree.expressions().forEach(expr -> {
        if (expr.is(Tree.Kind.NAME)) {
          addBindingUsage((PyNameTree) expr, Usage.Kind.LOOP_DECLARATION);
        }
      });
    }

    private void createParameters(PyFunctionLikeTree function) {
      PyParameterListTree parameterList = function.parameters();
      if (parameterList == null || parameterList.all().isEmpty()) {
        return;
      }

      boolean hasSelf = false;
      if (function.isMethodDefinition()) {
        PyAnyParameterTree first = parameterList.all().get(0);
        if (first.is(Kind.PARAMETER)) {
          currentScope().createSelfParameter((PyParameterTree) first);
          hasSelf = true;
        }
      }

      parameterList.nonTuple().stream()
        .skip(hasSelf ? 1 : 0)
        .forEach(param -> addBindingUsage(param.name(), Usage.Kind.PARAMETER));

      parameterList.all().stream()
        .filter(param -> param.is(Kind.TUPLE))
        .flatMap(param -> ((PyTupleTree) param).elements().stream())
        .filter(param -> param.is(Kind.NAME))
        .map(PyNameTree.class::cast)
        .forEach(name -> addBindingUsage(name, Usage.Kind.PARAMETER));
    }

    @Override
    public void visitAssignmentStatement(PyAssignmentStatementTree pyAssignmentStatementTree) {
      List<PyExpressionTree> lhs = pyAssignmentStatementTree.lhsExpressions().stream()
        .flatMap(exprList -> exprList.expressions().stream())
        .flatMap(this::flattenTuples)
        .collect(Collectors.toList());

      assignmentLeftHandSides.addAll(lhs);

      lhs.stream()
        .filter(expr -> expr.is(Kind.NAME))
        .map(PyNameTree.class::cast)
        .forEach(name -> addBindingUsage(name, Usage.Kind.ASSIGNMENT_LHS));

      super.visitAssignmentStatement(pyAssignmentStatementTree);
    }

    private Stream<PyExpressionTree> flattenTuples(PyExpressionTree expression) {
      if (expression.is(Kind.TUPLE)) {
        PyTupleTree tuple = (PyTupleTree) expression;
        return tuple.elements().stream().flatMap(this::flattenTuples);
      } else {
        return Stream.of(expression);
      }
    }

    @Override
    public void visitAnnotatedAssignment(PyAnnotatedAssignmentTree pyAnnotatedAssignmentTree) {
      if (pyAnnotatedAssignmentTree.variable().is(Kind.NAME)) {
        addBindingUsage((PyNameTree) pyAnnotatedAssignmentTree.variable(), Usage.Kind.ASSIGNMENT_LHS);
      }
      super.visitAnnotatedAssignment(pyAnnotatedAssignmentTree);
    }

    @Override
    public void visitCompoundAssignment(PyCompoundAssignmentStatementTree pyCompoundAssignmentStatementTree) {
      if (pyCompoundAssignmentStatementTree.lhsExpression().is(Kind.NAME)) {
        addBindingUsage((PyNameTree) pyCompoundAssignmentStatementTree.lhsExpression(), Usage.Kind.COMPOUND_ASSIGNMENT_LHS);
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


    private void addBindingUsage(PyNameTree nameTree, Usage.Kind usage, @Nullable String fullyQualifiedName) {
      currentScope().addBindingUsage(nameTree, usage, fullyQualifiedName);
    }

    private void addBindingUsage(PyNameTree nameTree, Usage.Kind usage) {
      currentScope().addBindingUsage(nameTree, usage, null);
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
    private final Map<String, SymbolImpl> instanceAttributesByName = new HashMap<>();

    private Scope(@Nullable Scope parent, Tree rootTree) {
      this.parent = parent;
      this.rootTree = rootTree;
    }

    private Set<TreeSymbol> symbols() {
      return Collections.unmodifiableSet(symbols);
    }

    private void createSelfParameter(PyParameterTree parameter) {
      PyNameTree nameTree = parameter.name();
      String symbolName = nameTree.name();
      SymbolImpl symbol = new SelfSymbolImpl(symbolName, parent);
      symbols.add(symbol);
      symbolsByName.put(symbolName, symbol);
      symbol.addUsage(nameTree, Usage.Kind.PARAMETER);
    }

    void addBindingUsage(PyNameTree nameTree, Usage.Kind kind, @Nullable String fullyQualifiedName) {
      String symbolName = nameTree.name();
      if (!symbolsByName.containsKey(symbolName) && !globalNames.contains(symbolName) && !nonlocalNames.contains(symbolName)) {
        SymbolImpl symbol = new SymbolImpl(symbolName, fullyQualifiedName);
        symbols.add(symbol);
        symbolsByName.put(symbolName, symbol);
      }
      SymbolImpl symbol = resolve(symbolName);
      if (symbol != null) {
        symbol.addUsage(nameTree, kind);
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
    @Nullable
    private String fullyQualifiedName;
    private final List<Usage> usages = new ArrayList<>();
    private Map<String, TreeSymbol> childrenSymbolByName = new HashMap<>();

    public SymbolImpl(String name, @Nullable String fullyQualifiedName) {
      this.name = name;
      this.fullyQualifiedName = fullyQualifiedName;
    }

    @Override
    public String name() {
      return name;
    }

    @Override
    public List<Usage> usages() {
      return Collections.unmodifiableList(usages);
    }

    @CheckForNull
    @Override
    public String fullyQualifiedName() {
      return fullyQualifiedName;
    }

    void addUsage(Tree tree, Usage.Kind kind) {
      usages.add(new UsageImpl(tree, kind));
      if (tree.is(Kind.NAME)) {
        ((PyNameTreeImpl) tree).setSymbol(this);
      }
       // we cannot know what is the fully qualified name (see FullyQualifiedNameTest#import_alias_reassigned)
      if (fullyQualifiedName != null && usages.stream().filter(Usage::isBindingUsage).count() > 1) {
        fullyQualifiedName = null;
      }
    }

    void addOrCreateChildUsage(PyNameTree name, Usage.Kind kind) {
      String childSymbolName = name.name();
      if (!childrenSymbolByName.containsKey(childSymbolName)) {
        String childFullyQualifiedName = fullyQualifiedName != null
          ? (fullyQualifiedName + "." + childSymbolName)
          : null;
        SymbolImpl symbol = new SymbolImpl(childSymbolName, childFullyQualifiedName);
        childrenSymbolByName.put(childSymbolName, symbol);
      }
      TreeSymbol symbol = childrenSymbolByName.get(childSymbolName);
      ((SymbolImpl) symbol).addUsage(name, kind);
    }
  }

  private static class SelfSymbolImpl extends SymbolImpl {

    private final Scope classScope;

    SelfSymbolImpl(String name, Scope classScope) {
      super(name, null);
      this.classScope = classScope;
    }

    @Override
    void addOrCreateChildUsage(PyNameTree nameTree, Usage.Kind kind) {
      SymbolImpl symbol = classScope.instanceAttributesByName.computeIfAbsent(nameTree.name(), name -> new SymbolImpl(name, null));
      symbol.addUsage(nameTree, kind);
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
    public void visitQualifiedExpression(PyQualifiedExpressionTree qualifiedExpression) {
      // We need to firstly create symbol for qualifier
      super.visitQualifiedExpression(qualifiedExpression);
      if (qualifiedExpression.qualifier() instanceof HasSymbol) {
        TreeSymbol qualifierSymbol = ((HasSymbol) qualifiedExpression.qualifier()).symbol();
        if (qualifierSymbol != null) {
          Usage.Kind usageKind = assignmentLeftHandSides.contains(qualifiedExpression) ? Usage.Kind.ASSIGNMENT_LHS : Usage.Kind.OTHER;
          ((SymbolImpl) qualifierSymbol).addOrCreateChildUsage(qualifiedExpression.name(), usageKind);
        }
      }
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
      // TODO: use Set to improve performances
      if (symbol != null && symbol.usages().stream().noneMatch(usage -> usage.tree().equals(nameTree))) {
        symbol.addUsage(nameTree, Usage.Kind.OTHER);
      }
    }
  }
}
