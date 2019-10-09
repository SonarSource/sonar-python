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
import org.sonar.python.api.tree.AliasedName;
import org.sonar.python.api.tree.AnnotatedAssignment;
import org.sonar.python.api.tree.AnyParameter;
import org.sonar.python.api.tree.AssignmentStatement;
import org.sonar.python.api.tree.ClassDef;
import org.sonar.python.api.tree.CompoundAssignmentStatement;
import org.sonar.python.api.tree.ComprehensionFor;
import org.sonar.python.api.tree.Decorator;
import org.sonar.python.api.tree.DottedName;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.ForStatement;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.FunctionLike;
import org.sonar.python.api.tree.GlobalStatement;
import org.sonar.python.api.tree.ImportFrom;
import org.sonar.python.api.tree.ImportName;
import org.sonar.python.api.tree.LambdaExpression;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.NonlocalStatement;
import org.sonar.python.api.tree.ParameterList;
import org.sonar.python.api.tree.Parameter;
import org.sonar.python.api.tree.QualifiedExpression;
import org.sonar.python.api.tree.Tuple;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.api.tree.TupleParameter;
import org.sonar.python.tree.BaseTreeVisitor;
import org.sonar.python.tree.ClassDefImpl;
import org.sonar.python.tree.FileInputImpl;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.LambdaExpressionImpl;
import org.sonar.python.tree.NameImpl;

// SymbolTable based on https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
public class SymbolTableBuilder extends BaseTreeVisitor {

  private Map<Tree, Scope> scopesByRootTree;
  private Set<Tree> assignmentLeftHandSides = new HashSet<>();

  @Override
  public void visitFileInput(FileInput fileInput) {
    scopesByRootTree = new HashMap<>();
    fileInput.accept(new FirstPhaseVisitor());
    fileInput.accept(new SecondPhaseVisitor());
    for (Scope scope : scopesByRootTree.values()) {
      if (scope.rootTree instanceof FunctionLike) {
        FunctionLike funcDef = (FunctionLike) scope.rootTree;
        for (Symbol symbol : scope.symbols()) {
          if (funcDef.is(Kind.LAMBDA)) {
            ((LambdaExpressionImpl) funcDef).addLocalVariableSymbol(symbol);
          } else {
            ((FunctionDefImpl) funcDef).addLocalVariableSymbol(symbol);
          }
        }
      } else if (scope.rootTree.is(Kind.CLASSDEF)) {
        ClassDefImpl classDef = (ClassDefImpl) scope.rootTree;
        scope.symbols.forEach(classDef::addClassField);
        scope.instanceAttributesByName.values().forEach(classDef::addInstanceField);
      } else if (scope.rootTree.is(Kind.FILE_INPUT)) {
        scope.symbols.forEach(((FileInputImpl) fileInput)::addGlobalVariables);
      }
    }
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
    public void visitFileInput(FileInput tree) {
      createScope(tree, null);
      enterScope(tree);
      super.visitFileInput(tree);
    }

    @Override
    public void visitLambda(LambdaExpression pyLambdaExpressionTree) {
      createScope(pyLambdaExpressionTree, currentScope());
      enterScope(pyLambdaExpressionTree);
      createParameters(pyLambdaExpressionTree);
      super.visitLambda(pyLambdaExpressionTree);
      leaveScope();
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      createScope(pyFunctionDefTree, currentScope());
      enterScope(pyFunctionDefTree);
      createParameters(pyFunctionDefTree);
      super.visitFunctionDef(pyFunctionDefTree);
      leaveScope();
    }

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      createScope(pyClassDefTree, currentScope());
      enterScope(pyClassDefTree);
      super.visitClassDef(pyClassDefTree);
      leaveScope();
    }

    @Override
    public void visitImportName(ImportName pyImportNameTree) {
      createImportedNames(pyImportNameTree.modules(), null, false);
      super.visitImportName(pyImportNameTree);
    }

    @Override
    public void visitImportFrom(ImportFrom pyImportFromTree) {
      DottedName moduleTree = pyImportFromTree.module();
      String moduleName = moduleTree != null
        ? moduleTree.names().stream().map(Name::name).collect(Collectors.joining("."))
        : null;
      createImportedNames(pyImportFromTree.importedNames(), moduleName, !pyImportFromTree.dottedPrefixForModule().isEmpty());
      super.visitImportFrom(pyImportFromTree);
    }

    private void createImportedNames(List<AliasedName> importedNames, @Nullable String fromModuleName, boolean isRelativeImport) {
      importedNames.forEach(module -> {
        Name nameTree = module.dottedName().names().get(0);
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
    public void visitForStatement(ForStatement pyForStatementTree) {
      createLoopVariables(pyForStatementTree);
      super.visitForStatement(pyForStatementTree);
    }

    @Override
    public void visitComprehensionFor(ComprehensionFor tree) {
      if (tree.loopExpression().is(Tree.Kind.NAME)) {
        addBindingUsage((Name) tree.loopExpression(), Usage.Kind.COMP_DECLARATION);
      }
      super.visitComprehensionFor(tree);
    }

    private void createLoopVariables(ForStatement loopTree) {
      loopTree.expressions().stream().flatMap(this::flattenTuples).forEach(expr -> {
        if (expr.is(Tree.Kind.NAME)) {
          addBindingUsage((Name) expr, Usage.Kind.LOOP_DECLARATION);
        }
      });
    }

    private void createParameters(FunctionLike function) {
      ParameterList parameterList = function.parameters();
      if (parameterList == null || parameterList.all().isEmpty()) {
        return;
      }

      boolean hasSelf = false;
      if (function.isMethodDefinition()) {
        AnyParameter first = parameterList.all().get(0);
        if (first.is(Kind.PARAMETER)) {
          currentScope().createSelfParameter((Parameter) first);
          hasSelf = true;
        }
      }

      parameterList.nonTuple()
        .stream()
        .skip(hasSelf ? 1 : 0)
        .forEach(param -> addBindingUsage(param.name(), Usage.Kind.PARAMETER));

      parameterList.all().stream()
        .filter(param -> param.is(Kind.TUPLE_PARAMETER))
        .map(TupleParameter.class::cast)
        .forEach(this::addTupleParamElementsToBindingUsage);
    }

    private void addTupleParamElementsToBindingUsage(TupleParameter param) {
      param.parameters().stream()
        .filter(p -> p.is(Kind.PARAMETER))
        .map(p -> ((Parameter) p).name())
        .forEach(name -> addBindingUsage(name, Usage.Kind.PARAMETER));
      param.parameters().stream()
        .filter(p -> p.is(Kind.TUPLE_PARAMETER))
        .map(TupleParameter.class::cast)
        .forEach(this::addTupleParamElementsToBindingUsage);
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement pyAssignmentStatementTree) {
      List<Expression> lhs = pyAssignmentStatementTree.lhsExpressions().stream()
        .flatMap(exprList -> exprList.expressions().stream())
        .flatMap(this::flattenTuples)
        .collect(Collectors.toList());

      assignmentLeftHandSides.addAll(lhs);

      lhs.stream()
        .filter(expr -> expr.is(Kind.NAME))
        .map(Name.class::cast)
        .forEach(name -> addBindingUsage(name, Usage.Kind.ASSIGNMENT_LHS));

      super.visitAssignmentStatement(pyAssignmentStatementTree);
    }

    private Stream<Expression> flattenTuples(Expression expression) {
      if (expression.is(Kind.TUPLE)) {
        Tuple tuple = (Tuple) expression;
        return tuple.elements().stream().flatMap(this::flattenTuples);
      } else {
        return Stream.of(expression);
      }
    }

    @Override
    public void visitAnnotatedAssignment(AnnotatedAssignment annotatedAssignment) {
      if (annotatedAssignment.variable().is(Kind.NAME)) {
        addBindingUsage((Name) annotatedAssignment.variable(), Usage.Kind.ASSIGNMENT_LHS);
      }
      super.visitAnnotatedAssignment(annotatedAssignment);
    }

    @Override
    public void visitCompoundAssignment(CompoundAssignmentStatement pyCompoundAssignmentStatementTree) {
      if (pyCompoundAssignmentStatementTree.lhsExpression().is(Kind.NAME)) {
        addBindingUsage((Name) pyCompoundAssignmentStatementTree.lhsExpression(), Usage.Kind.COMPOUND_ASSIGNMENT_LHS);
      }
      super.visitCompoundAssignment(pyCompoundAssignmentStatementTree);
    }

    @Override
    public void visitGlobalStatement(GlobalStatement pyGlobalStatementTree) {
      pyGlobalStatementTree.variables().stream()
        .map(Name::name)
        .forEach(name -> currentScope().addGlobalName(name));
      super.visitGlobalStatement(pyGlobalStatementTree);
    }

    @Override
    public void visitNonlocalStatement(NonlocalStatement pyNonlocalStatementTree) {
      pyNonlocalStatementTree.variables().stream()
        .map(Name::name)
        .forEach(name -> currentScope().addNonLocalName(name));
      super.visitNonlocalStatement(pyNonlocalStatementTree);
    }

    private void createScope(Tree tree, @Nullable Scope parent) {
      scopesByRootTree.put(tree, new Scope(parent, tree));
    }


    private void addBindingUsage(Name nameTree, Usage.Kind usage, @Nullable String fullyQualifiedName) {
      currentScope().addBindingUsage(nameTree, usage, fullyQualifiedName);
    }

    private void addBindingUsage(Name nameTree, Usage.Kind usage) {
      currentScope().addBindingUsage(nameTree, usage, null);
    }

    private Scope currentScope() {
      return scopesByRootTree.get(currentScopeRootTree());
    }

  }

  private static class Scope {

    private final Tree rootTree;
    private final Scope parent;
    private final Map<String, Symbol> symbolsByName = new HashMap<>();
    private final Set<Symbol> symbols = new HashSet<>();
    private final Set<String> globalNames = new HashSet<>();
    private final Set<String> nonlocalNames = new HashSet<>();
    private final Map<String, SymbolImpl> instanceAttributesByName = new HashMap<>();

    private Scope(@Nullable Scope parent, Tree rootTree) {
      this.parent = parent;
      this.rootTree = rootTree;
    }

    private Set<Symbol> symbols() {
      return Collections.unmodifiableSet(symbols);
    }

    private void createSelfParameter(Parameter parameter) {
      Name nameTree = parameter.name();
      String symbolName = nameTree.name();
      SymbolImpl symbol = new SelfSymbolImpl(symbolName, parent);
      symbols.add(symbol);
      symbolsByName.put(symbolName, symbol);
      symbol.addUsage(nameTree, Usage.Kind.PARAMETER);
    }

    void addBindingUsage(Name nameTree, Usage.Kind kind, @Nullable String fullyQualifiedName) {
      String symbolName = nameTree.name();
      if (!symbolsByName.containsKey(symbolName) && !globalNames.contains(symbolName) && !nonlocalNames.contains(symbolName)) {
        SymbolImpl symbol = new SymbolImpl(symbolName, fullyQualifiedName);
        symbols.add(symbol);
        symbolsByName.put(symbolName, symbol);
      }
      SymbolImpl symbol = resolve(symbolName);
      if (symbol != null) {
        if (fullyQualifiedName != null && !fullyQualifiedName.equals(symbol.fullyQualifiedName)) {
          symbol.fullyQualifiedName = null;
        }
        if (fullyQualifiedName == null && symbol.fullyQualifiedName != null) {
          symbol.fullyQualifiedName = null;
        }
        symbol.addUsage(nameTree, kind);
      }
    }

    @CheckForNull
    SymbolImpl resolve(String symbolName) {
      Symbol symbol = symbolsByName.get(symbolName);
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

  private static class SymbolImpl implements Symbol {

    private final String name;
    @Nullable
    private String fullyQualifiedName;
    private final List<Usage> usages = new ArrayList<>();
    private Map<String, Symbol> childrenSymbolByName = new HashMap<>();

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
        ((NameImpl) tree).setSymbol(this);
      }
    }

    void addOrCreateChildUsage(Name name, Usage.Kind kind) {
      String childSymbolName = name.name();
      if (!childrenSymbolByName.containsKey(childSymbolName)) {
        String childFullyQualifiedName = fullyQualifiedName != null
          ? (fullyQualifiedName + "." + childSymbolName)
          : null;
        SymbolImpl symbol = new SymbolImpl(childSymbolName, childFullyQualifiedName);
        childrenSymbolByName.put(childSymbolName, symbol);
      }
      Symbol symbol = childrenSymbolByName.get(childSymbolName);
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
    void addOrCreateChildUsage(Name nameTree, Usage.Kind kind) {
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
    public void visitFileInput(FileInput tree) {
      enterScope(tree);
      super.visitFileInput(tree);
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      enterScope(pyFunctionDefTree);
      super.visitFunctionDef(pyFunctionDefTree);
      leaveScope();
    }

    @Override
    public void visitLambda(LambdaExpression pyLambdaExpressionTree) {
      enterScope(pyLambdaExpressionTree);
      super.visitLambda(pyLambdaExpressionTree);
      leaveScope();
    }

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      enterScope(pyClassDefTree);
      super.visitClassDef(pyClassDefTree);
      leaveScope();
    }

    @Override
    public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
      // We need to firstly create symbol for qualifier
      super.visitQualifiedExpression(qualifiedExpression);
      if (qualifiedExpression.qualifier() instanceof HasSymbol) {
        Symbol qualifierSymbol = ((HasSymbol) qualifiedExpression.qualifier()).symbol();
        if (qualifierSymbol != null) {
          Usage.Kind usageKind = assignmentLeftHandSides.contains(qualifiedExpression) ? Usage.Kind.ASSIGNMENT_LHS : Usage.Kind.OTHER;
          ((SymbolImpl) qualifierSymbol).addOrCreateChildUsage(qualifiedExpression.name(), usageKind);
        }
      }
    }

    @Override
    public void visitDecorator(Decorator decorator) {
      Name nameTree = decorator.name().names().get(0);
      addSymbolUsage(nameTree);
      super.visitDecorator(decorator);
    }

    @Override
    public void visitName(Name pyNameTree) {
      if (!pyNameTree.isVariable()) {
        return;
      }
      addSymbolUsage(pyNameTree);
      super.visitName(pyNameTree);
    }

    private void addSymbolUsage(Name nameTree) {
      Scope scope = scopesByRootTree.get(currentScopeRootTree());
      SymbolImpl symbol = scope.resolve(nameTree.name());
      // TODO: use Set to improve performances
      if (symbol != null && symbol.usages().stream().noneMatch(usage -> usage.tree().equals(nameTree))) {
        symbol.addUsage(nameTree, Usage.Kind.OTHER);
      }
    }
  }
}
