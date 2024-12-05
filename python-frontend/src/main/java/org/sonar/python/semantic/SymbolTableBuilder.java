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
package org.sonar.python.semantic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CapturePattern;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.FunctionLike;
import org.sonar.plugins.python.api.tree.GlobalStatement;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NonlocalStatement;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.TupleParameter;
import org.sonar.plugins.python.api.tree.TypeAliasStatement;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.tree.TypeParams;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.python.tree.ClassDefImpl;
import org.sonar.python.tree.ComprehensionExpressionImpl;
import org.sonar.python.tree.DictCompExpressionImpl;
import org.sonar.python.tree.FileInputImpl;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.ImportFromImpl;
import org.sonar.python.tree.LambdaExpressionImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeInference;
import org.sonar.python.types.TypeShed;

import static org.sonar.python.semantic.SymbolUtils.boundNamesFromExpression;
import static org.sonar.python.semantic.SymbolUtils.resolveTypeHierarchy;

// SymbolTable based on https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
public class SymbolTableBuilder extends BaseTreeVisitor {
  private final String fullyQualifiedModuleName;
  private final List<String> filePath;
  private final ProjectLevelSymbolTable projectLevelSymbolTable;
  private Map<Tree, Scope> scopesByRootTree;
  private FileInput fileInput = null;
  private final Set<Tree> assignmentLeftHandSides = new HashSet<>();
  private final PythonFile pythonFile;
  private final Set<String> importedModulesFQN = new HashSet<>();

  public SymbolTableBuilder(PythonFile pythonFile) {
    fullyQualifiedModuleName = null;
    filePath = null;
    projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    this.pythonFile = pythonFile;
  }

  public Set<String> importedModulesFQN() {
    return Collections.unmodifiableSet(importedModulesFQN);
  }

  public SymbolTableBuilder(String packageName, PythonFile pythonFile) {
    this(packageName, pythonFile, ProjectLevelSymbolTable.empty());
  }

  public SymbolTableBuilder(String packageName, PythonFile pythonFile, ProjectLevelSymbolTable projectLevelSymbolTable) {
    this.pythonFile = pythonFile;
    String fileName = pythonFile.fileName();
    fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, fileName);
    filePath = new ArrayList<>(Arrays.asList(fullyQualifiedModuleName.split("\\.")));
    if (SymbolUtils.getModuleFileName(fileName).equals("__init__")) {
      filePath.add("");
    }
    this.projectLevelSymbolTable = projectLevelSymbolTable;
  }

  @Override
  public void visitFileInput(FileInput fileInput) {
    this.fileInput = fileInput;
    scopesByRootTree = new HashMap<>();
    fileInput.accept(new FirstPhaseVisitor());
    fileInput.accept(new SecondPhaseVisitor());
    createAmbiguousSymbols();
    addSymbolsToTree((FileInputImpl) fileInput);
    fileInput.accept(new ThirdPhaseVisitor());
    TypeInference.inferTypes(fileInput, pythonFile);
  }

  private static class SymbolToUpdate {
    final Symbol symbol;
    final AmbiguousSymbol ambiguousSymbol;

    SymbolToUpdate(Symbol symbol, AmbiguousSymbol ambiguousSymbol) {
      this.symbol = symbol;
      this.ambiguousSymbol = ambiguousSymbol;
    }
  }

  private void createAmbiguousSymbols() {
    for (Scope scope : scopesByRootTree.values()) {
      Set<SymbolToUpdate> symbolsToUpdate = new HashSet<>();
      for (Symbol symbol : scope.symbols()) {
        if (symbol.kind() == Symbol.Kind.OTHER) {
          List<Usage> bindingUsages = symbol.usages().stream().filter(Usage::isBindingUsage).toList();
          if (bindingUsages.size() > 1 &&
            bindingUsages.stream().anyMatch(usage -> usage.kind() == Usage.Kind.FUNC_DECLARATION || usage.kind() == Usage.Kind.CLASS_DECLARATION)) {

            Set<Symbol> alternativeDefinitions = getAlternativeDefinitions(symbol, bindingUsages);
            AmbiguousSymbol ambiguousSymbol = AmbiguousSymbolImpl.create(alternativeDefinitions);
            // update symbol and usage to newly created ambiguous symbol
            symbol.usages().forEach(usage -> ((SymbolImpl) ambiguousSymbol).addUsage(usage.tree(), usage.kind()));
            symbolsToUpdate.add(new SymbolToUpdate(symbol, ambiguousSymbol));
          }
        }
      }
      symbolsToUpdate.forEach(symbolToUpdate -> scope.replaceSymbolWithAmbiguousSymbol(symbolToUpdate.symbol, symbolToUpdate.ambiguousSymbol));
    }
  }

  private Set<Symbol> getAlternativeDefinitions(Symbol symbol, List<Usage> bindingUsages) {
    Set<Symbol> alternativeDefinitions = new HashSet<>();
    for (Usage bindingUsage : bindingUsages) {
      switch (bindingUsage.kind()) {
        case FUNC_DECLARATION:
          FunctionDef functionDef = (FunctionDef) bindingUsage.tree().parent();
          FunctionSymbolImpl functionSymbol = new FunctionSymbolImpl(functionDef, symbol.fullyQualifiedName(), pythonFile);
          ((FunctionDefImpl) functionDef).setFunctionSymbol(functionSymbol);
          alternativeDefinitions.add(functionSymbol);
          break;
        case CLASS_DECLARATION:
          ClassDef classDef = (ClassDef) bindingUsage.tree().parent();
          ClassSymbolImpl classSymbol = new ClassSymbolImpl(classDef, symbol.fullyQualifiedName(), pythonFile);
          resolveTypeHierarchy(classDef, classSymbol, pythonFile, scopesByRootTree.get(fileInput).symbolsByName);
          Scope classScope = scopesByRootTree.get(classDef);
          classSymbol.addMembers(getClassMembers(classScope.symbolsByName, classScope.instanceAttributesByName));
          alternativeDefinitions.add(classSymbol);
          break;
        default:
          SymbolImpl alternativeSymbol = new SymbolImpl(symbol.name(), symbol.fullyQualifiedName());
          alternativeDefinitions.add(alternativeSymbol);
      }
    }
    return alternativeDefinitions;
  }

  private void addSymbolsToTree(FileInputImpl fileInput) {
    for (Scope scope : scopesByRootTree.values()) {
      if (scope.rootTree instanceof FunctionLike funcDef) {
        for (Symbol symbol : scope.symbols()) {
          if (funcDef.is(Kind.LAMBDA)) {
            ((LambdaExpressionImpl) funcDef).addLocalVariableSymbol(symbol);
          } else {
            ((FunctionDefImpl) funcDef).addLocalVariableSymbol(symbol);
          }
        }
      } else if (scope.rootTree.is(Kind.CLASSDEF)) {
        ClassDefImpl classDef = (ClassDefImpl) scope.rootTree;
        scope.symbols().forEach(classDef::addClassField);
        scope.instanceAttributesByName.values().forEach(classDef::addInstanceField);
        Symbol classSymbol = classDef.name().symbol();
        Optional.ofNullable(classSymbol)
          .filter(symbol -> symbol.kind() == Symbol.Kind.CLASS)
          .map(ClassSymbolImpl.class::cast)
          .ifPresent(symbol -> symbol.addMembers(getClassMembers(scope.symbolsByName, scope.instanceAttributesByName)));

      } else if (scope.rootTree.is(Kind.FILE_INPUT)) {
        scope.symbols().stream().filter(s -> !scope.builtinSymbols.contains(s)).forEach(fileInput::addGlobalVariables);
      } else if (scope.rootTree.is(Kind.DICT_COMPREHENSION)) {
        scope.symbols().forEach(((DictCompExpressionImpl) scope.rootTree)::addLocalVariableSymbol);
      } else if (scope.rootTree instanceof ComprehensionExpression) {
        scope.symbols().forEach(((ComprehensionExpressionImpl) scope.rootTree)::addLocalVariableSymbol);
      }
    }
  }

  private static Set<Symbol> getClassMembers(Map<String, Symbol> symbolsInClass, Map<String, SymbolImpl> instanceAttributesByName) {
    Set<Symbol> members = new HashSet<>(symbolsInClass.values());
    for (SymbolImpl instanceAttribute : instanceAttributesByName.values()) {
      SymbolImpl member = (SymbolImpl) symbolsInClass.get(instanceAttribute.name());
      if (member != null) {
        for (Usage usage : instanceAttribute.usages()) {
          if (usage.isBindingUsage()) {
            // TODO: should be an Ambiguous Symbol
            member.setKind(Symbol.Kind.OTHER);
          }
          member.addUsage(usage.tree(), usage.kind());
        }
      } else {
        members.add(instanceAttribute);
      }
    }
    return members;
  }

  private class ScopeVisitor extends BaseTreeVisitor {

    private final Deque<Tree> scopeRootTrees = new LinkedList<>();
    protected Scope moduleScope;

    Tree currentScopeRootTree() {
      return scopeRootTrees.peek();
    }

    void enterScope(Tree tree) {
      scopeRootTrees.push(tree);
    }

    Tree leaveScope() {
      return scopeRootTrees.pop();
    }

    Scope currentScope() {
      return scopesByRootTree.get(currentScopeRootTree());
    }
  }

  private class FirstPhaseVisitor extends ScopeVisitor {

    @Override
    public void visitFileInput(FileInput tree) {
      createScope(tree, null);
      enterScope(tree);
      moduleScope = currentScope();
      Map<String, Symbol> typeShedSymbols = TypeShed.builtinSymbols();
      for (String name : BuiltinSymbols.all()) {
        currentScope().createBuiltinSymbol(name, typeShedSymbols);
      }
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
    public void visitDictCompExpression(DictCompExpression tree) {
      createScope(tree, currentScope());
      enterScope(tree);
      super.visitDictCompExpression(tree);
      leaveScope();
    }

    /**
     * The scope of the decorator should be the parent scope of the function or class to which the decorator is assigned.
     * So we have to leave the function or class scope, visit the decorator and enter the previous scope.
     */
    @Override
    public void visitDecorator(Decorator tree) {
      leaveScope();
      super.visitDecorator(tree);
      enterScope(tree.parent());
    }

    @Override
    public void visitPyListOrSetCompExpression(ComprehensionExpression tree) {
      createScope(tree, currentScope());
      enterScope(tree);
      super.visitPyListOrSetCompExpression(tree);
      leaveScope();
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      String functionName = pyFunctionDefTree.name().name();
      String fullyQualifiedName = getFullyQualifiedName(functionName);
      currentScope().addFunctionSymbol(pyFunctionDefTree, fullyQualifiedName);
      createScope(pyFunctionDefTree, currentScope());
      enterScope(pyFunctionDefTree);
      createTypeParameters(pyFunctionDefTree.typeParams());
      createParameters(pyFunctionDefTree);
      super.visitFunctionDef(pyFunctionDefTree);
      leaveScope();
    }

    private void createTypeParameters(@Nullable TypeParams typeParams) {
      Optional.ofNullable(typeParams)
        .map(TypeParams::typeParamsList)
        .stream()
        .flatMap(Collection::stream)
        .forEach(typeParam -> addBindingUsage(typeParam.name(), Usage.Kind.TYPE_PARAM_DECLARATION));
    }

    @Override
    public void visitTypeAliasStatement(TypeAliasStatement typeAliasStatement) {
      addBindingUsage(typeAliasStatement.name(), Usage.Kind.TYPE_ALIAS_DECLARATION);
      super.visitTypeAliasStatement(typeAliasStatement);
    }

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      String className = pyClassDefTree.name().name();
      String fullyQualifiedName = getFullyQualifiedName(className);
      currentScope().addClassSymbol(pyClassDefTree, fullyQualifiedName);
      createScope(pyClassDefTree, currentScope());
      enterScope(pyClassDefTree);
      createTypeParameters(pyClassDefTree.typeParams());
      super.visitClassDef(pyClassDefTree);
      leaveScope();
    }

    @CheckForNull
    private String getFullyQualifiedName(String name) {
      String prefix = scopeQualifiedName();
      if (prefix != null) {
        return prefix.isEmpty() ? name : (prefix + "." + name);
      }
      return null;
    }

    private String scopeQualifiedName() {
      Tree scopeTree = currentScopeRootTree();
      if (scopeTree.is(Kind.CLASSDEF, Kind.FUNCDEF)) {
        Name name = scopeTree.is(Kind.CLASSDEF)
          ? ((ClassDef) scopeTree).name()
          : ((FunctionDef) scopeTree).name();
        return Optional.ofNullable(name.symbol()).map(Symbol::fullyQualifiedName).orElse(name.name());
      }
      return fullyQualifiedModuleName;
    }

    @Override
    public void visitImportName(ImportName pyImportNameTree) {
      createImportedNames(pyImportNameTree.modules(), null, Collections.emptyList());
      super.visitImportName(pyImportNameTree);
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      DottedName moduleTree = importFrom.module();
      String moduleName = moduleTree != null
        ? moduleTree.names().stream().map(Name::name).collect(Collectors.joining("."))
        : null;
      if (importFrom.isWildcardImport()) {
        importedModulesFQN.add(moduleName);
        Set<Symbol> importedModuleSymbols = projectLevelSymbolTable.getSymbolsFromModule(moduleName);
        if (importedModuleSymbols == null && moduleName != null && !moduleName.equals(fullyQualifiedModuleName)) {
          importedModuleSymbols = TypeShed.symbolsForModule(moduleName).values().stream()
            .map(importedSymbol -> currentScope().copySymbol(importedSymbol.name(), importedSymbol)).collect(Collectors.toSet());
        }
        if (importedModuleSymbols != null && !importedModuleSymbols.isEmpty()) {
          currentScope().createSymbolsFromWildcardImport(importedModuleSymbols, importFrom);
          ((ImportFromImpl) importFrom).setHasUnresolvedWildcardImport(false);
        } else {
          ((ImportFromImpl) importFrom).setHasUnresolvedWildcardImport(true);
        }
      } else {
        createImportedNames(importFrom.importedNames(), moduleName, importFrom.dottedPrefixForModule());
      }
      super.visitImportFrom(importFrom);
    }

    private void createImportedNames(List<AliasedName> importedNames, @Nullable String fromModuleName, List<Token> dottedPrefix) {
      importedNames.forEach(module -> {
        List<Name> dottedNames = module.dottedName().names();
        Name nameTree = dottedNames.get(0);
        String targetModuleName = fromModuleName;
        String fullyQualifiedName = targetModuleName != null
          ? (targetModuleName + "." + nameTree.name())
          : nameTree.name();
        if (!dottedPrefix.isEmpty()) {
          fullyQualifiedName = resolveFullyQualifiedNameBasedOnRelativeImport(dottedPrefix, fullyQualifiedName);
          targetModuleName = resolveFullyQualifiedNameBasedOnRelativeImport(dottedPrefix, targetModuleName);
        } else {
          importedModulesFQN.add(fullyQualifiedName);
        }
        Name alias = module.alias();
        if (targetModuleName != null) {
          currentScope().addImportedSymbol(alias == null ? nameTree : alias, fullyQualifiedName, targetModuleName);
        } else if (alias != null) {
          String fullName = dottedNames.stream().map(Name::name).collect(Collectors.joining("."));
          currentScope().addModuleSymbol(alias, fullName);
        } else if (dottedPrefix.isEmpty() && dottedNames.size() > 1) {
            // Submodule import
          dottedNames.stream().map(Name::name).reduce((a, b) -> String.join(".", a, b))
            .ifPresent(fqn -> currentScope().addSubmoduleSymbol(nameTree, fqn));
        } else {
          // It's a simple case - no "from" imports or aliasing
          currentScope().addModuleSymbol(nameTree, fullyQualifiedName);
        }
      });
    }

    @CheckForNull
    private String resolveFullyQualifiedNameBasedOnRelativeImport(List<Token> dottedPrefix, @Nullable String moduleName) {
      if (filePath == null || dottedPrefix.size() > filePath.size()) {
        return null;
      }
      String resolvedPackageName = String.join(".", filePath.subList(0, filePath.size() - dottedPrefix.size()));
      if (moduleName == null) {
        importedModulesFQN.add(resolvedPackageName);
        return resolvedPackageName;
      }
      if (resolvedPackageName.isEmpty()) {
        importedModulesFQN.add(moduleName);
        return moduleName;
      }
      String resolvedModuleFQN = resolvedPackageName + "." + moduleName;
      importedModulesFQN.add(resolvedModuleFQN);
      return resolvedModuleFQN;
    }

    @Override
    public void visitForStatement(ForStatement pyForStatementTree) {
      createLoopVariables(pyForStatementTree);
      super.visitForStatement(pyForStatementTree);
    }

    @Override
    public void visitComprehensionFor(ComprehensionFor tree) {
      addCompDeclarationParam(tree.loopExpression());
      super.visitComprehensionFor(tree);
    }

    private void addCompDeclarationParam(Tree tree) {
      boundNamesFromExpression(tree).forEach(name -> addBindingUsage(name, Usage.Kind.COMP_DECLARATION));
    }

    private void createLoopVariables(ForStatement loopTree) {
      loopTree.expressions().forEach(expr ->
        boundNamesFromExpression(expr).forEach(name -> addBindingUsage(name, Usage.Kind.LOOP_DECLARATION)));
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
        .map(Parameter::name)
        .filter(Objects::nonNull)
        .forEach(param -> addBindingUsage(param, Usage.Kind.PARAMETER));

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
      List<Expression> lhs = SymbolUtils.assignmentsLhs(pyAssignmentStatementTree);

      assignmentLeftHandSides.addAll(lhs);

      lhs.forEach(expression -> boundNamesFromExpression(expression).forEach(name -> addBindingUsage(name, Usage.Kind.ASSIGNMENT_LHS)));

      super.visitAssignmentStatement(pyAssignmentStatementTree);
    }

    @Override
    public void visitAnnotatedAssignment(AnnotatedAssignment annotatedAssignment) {
      if (annotatedAssignment.variable().is(Kind.NAME)) {
        Name variable = (Name) annotatedAssignment.variable();
        addBindingUsage(variable, Usage.Kind.ASSIGNMENT_LHS, getFullyQualifiedName(variable.name()));
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
    public void visitAssignmentExpression(AssignmentExpression assignmentExpression) {
      final Scope scope = currentScope();
      Tree scopeRootTree = scope.rootTree;
      if (scopeRootTree.is(Kind.GENERATOR_EXPR) || scopeRootTree instanceof ComprehensionExpression || scopeRootTree instanceof DictCompExpression) {
        scope.parent().addBindingUsage(assignmentExpression.lhsName(), Usage.Kind.ASSIGNMENT_LHS, null);
      } else {
        addBindingUsage(assignmentExpression.lhsName(), Usage.Kind.ASSIGNMENT_LHS);
      }
      super.visitAssignmentExpression(assignmentExpression);
    }

    @Override
    public void visitGlobalStatement(GlobalStatement pyGlobalStatementTree) {
      pyGlobalStatementTree.variables()
        .forEach(name -> {
          // Global statements are not binding usages, but we consider them as such for symbol creation
          moduleScope.addBindingUsage(name, Usage.Kind.GLOBAL_DECLARATION, null);
          currentScope().addGlobalName(name.name());
        });

      super.visitGlobalStatement(pyGlobalStatementTree);
    }

    @Override
    public void visitNonlocalStatement(NonlocalStatement pyNonlocalStatementTree) {
      pyNonlocalStatementTree.variables().stream()
        .map(Name::name)
        .forEach(name -> currentScope().addNonLocalName(name));
      super.visitNonlocalStatement(pyNonlocalStatementTree);
    }

    @Override
    public void visitExceptClause(ExceptClause exceptClause) {
      boundNamesFromExpression(exceptClause.exceptionInstance()).forEach(name -> addBindingUsage(name, Usage.Kind.EXCEPTION_INSTANCE));
      super.visitExceptClause(exceptClause);
    }

    @Override
    public void visitWithItem(WithItem withItem) {
      boundNamesFromExpression(withItem.expression()).forEach(name -> addBindingUsage(name, Usage.Kind.WITH_INSTANCE));
      super.visitWithItem(withItem);
    }

    @Override
    public void visitCapturePattern(CapturePattern capturePattern) {
      addBindingUsage(capturePattern.name(), Usage.Kind.PATTERN_DECLARATION);
      super.visitCapturePattern(capturePattern);
    }

    private void createScope(Tree tree, @Nullable Scope parent) {
      scopesByRootTree.put(tree, new Scope(parent, tree, pythonFile, fullyQualifiedModuleName, projectLevelSymbolTable));
    }

    private void addBindingUsage(Name nameTree, Usage.Kind usage) {
      addBindingUsage(nameTree, usage, null);
    }

    private void addBindingUsage(Name nameTree, Usage.Kind usage, @Nullable String fullyQualifiedName) {
      currentScope().addBindingUsage(nameTree, usage, fullyQualifiedName);
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
      scan(pyFunctionDefTree.decorators());
      enterScope(pyFunctionDefTree);
      scan(pyFunctionDefTree.name());
      scan(pyFunctionDefTree.typeParams());
      scan(pyFunctionDefTree.parameters());
      scan(pyFunctionDefTree.returnTypeAnnotation());
      scan(pyFunctionDefTree.body());
      leaveScope();
    }

    @Override
    public void visitParameter(Parameter parameter) {
      // parameter default value should not be in the function scope.
      Tree currentScopeTree = leaveScope();
      scan(parameter.defaultValue());
      enterScope(currentScopeTree);
      scan(parameter.name());
      scan(parameter.typeAnnotation());
    }

    @Override
    public void visitLambda(LambdaExpression pyLambdaExpressionTree) {
      enterScope(pyLambdaExpressionTree);
      super.visitLambda(pyLambdaExpressionTree);
      leaveScope();
    }

    @Override
    public void visitPyListOrSetCompExpression(ComprehensionExpression tree) {
      enterScope(tree);
      scan(tree.resultExpression());
      ComprehensionFor comprehensionFor = tree.comprehensionFor();
      scan(comprehensionFor.loopExpression());
      leaveScope();
      scan(comprehensionFor.iterable());
      enterScope(tree);
      scan(comprehensionFor.nestedClause());
      leaveScope();
    }

    @Override
    public void visitDictCompExpression(DictCompExpression tree) {
      enterScope(tree);
      scan(tree.keyExpression());
      scan(tree.valueExpression());
      ComprehensionFor comprehensionFor = tree.comprehensionFor();
      scan(comprehensionFor.loopExpression());
      leaveScope();
      scan(comprehensionFor.iterable());
      enterScope(tree);
      scan(comprehensionFor.nestedClause());
      leaveScope();
    }

    @Override
    public void visitTypeAnnotation(TypeAnnotation tree) {
      if (tree.is(Kind.PARAMETER_TYPE_ANNOTATION) || tree.is(Kind.RETURN_TYPE_ANNOTATION)) {
        // The scope of the type annotations on a function declaration should be the scope enclosing the function, and not the scope of
        // the function body itself. Note that this code assumes that we already entered the function body scope by visiting the type
        // annotations, so there should always be a scope to pop out and return to here.
        Tree currentScopeTree = leaveScope();
        super.visitTypeAnnotation(tree);
        enterScope(currentScopeTree);
        super.visitTypeAnnotation(tree);
      } else {
        super.visitTypeAnnotation(tree);
      }
    }

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      scan(pyClassDefTree.args());
      scan(pyClassDefTree.decorators());
      enterScope(pyClassDefTree);
      scan(pyClassDefTree.name());
      resolveTypeHierarchy(pyClassDefTree, pyClassDefTree.name().symbol(), pythonFile, scopesByRootTree.get(fileInput).symbolsByName);
      scan(pyClassDefTree.body());
      leaveScope();
    }

    @Override
    public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
      // We need to firstly create symbol for qualifier
      super.visitQualifiedExpression(qualifiedExpression);
      if (qualifiedExpression.qualifier() instanceof HasSymbol hasSymbol) {
        Symbol qualifierSymbol = hasSymbol.symbol();
        if (qualifierSymbol != null) {
          Usage.Kind usageKind = assignmentLeftHandSides.contains(qualifiedExpression) ? Usage.Kind.ASSIGNMENT_LHS : Usage.Kind.OTHER;
          ((SymbolImpl) qualifierSymbol).addOrCreateChildUsage(qualifiedExpression.name(), usageKind);
        }
      }
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

  private class ThirdPhaseVisitor extends BaseTreeVisitor {

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      FunctionSymbol functionSymbol = ((FunctionDefImpl) functionDef).functionSymbol();
      ParameterList parameters = functionDef.parameters();
      if (functionSymbol != null) {
        FunctionSymbolImpl functionSymbolImpl = (FunctionSymbolImpl) functionSymbol;
        if (parameters != null) {
          functionSymbolImpl.setParametersWithType(parameters);
        }
        TypeAnnotation typeAnnotation = functionDef.returnTypeAnnotation();
        if (typeAnnotation != null) {
          functionSymbolImpl.setDeclaredReturnType(InferredTypes.fromTypeAnnotation(typeAnnotation));
        }
      }
      super.visitFunctionDef(functionDef);
    }

    @Override
    public void visitAnnotatedAssignment(AnnotatedAssignment annotatedAssignment) {
      if (annotatedAssignment.variable().is(Kind.NAME)) {
        Name variable = (Name) annotatedAssignment.variable();
        TypeAnnotation annotation = annotatedAssignment.annotation();
        Symbol symbol = variable.symbol();
        Optional.ofNullable(symbol)
          .ifPresent(s -> ((SymbolImpl) symbol).setInferredType(InferredTypes.fromTypeAnnotation(annotation)));
      }
      super.visitAnnotatedAssignment(annotatedAssignment);
    }

    /**
     * Handle class member usages like the following:
     * <pre>
     *     class A:
     *       foo = 42
     *     print(A.foo)
     * </pre>
     */
    @Override
    public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
      super.visitQualifiedExpression(qualifiedExpression);
      TreeUtils.getSymbolFromTree(qualifiedExpression.qualifier())
        .filter(symbol -> symbol.kind() == Symbol.Kind.CLASS)
        .map(ClassSymbol.class::cast)
        .flatMap(classSymbol -> classSymbol.resolveMember(qualifiedExpression.name().name()))
        .ifPresent(member -> {
          Usage.Kind usageKind = assignmentLeftHandSides.contains(qualifiedExpression) ? Usage.Kind.ASSIGNMENT_LHS : Usage.Kind.OTHER;
          ((SymbolImpl) member).addUsage(qualifiedExpression.name(), usageKind);
        });
    }
  }
}
