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

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CapturePattern;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.FunctionLike;
import org.sonar.plugins.python.api.tree.GlobalStatement;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NonlocalStatement;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TupleParameter;
import org.sonar.plugins.python.api.tree.TypeAliasStatement;
import org.sonar.plugins.python.api.tree.TypeParams;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.python.semantic.SymbolUtils;

import static org.sonar.python.semantic.SymbolUtils.boundNamesFromExpression;

public class WriteUsagesVisitor extends ScopeVisitor {

  private ScopeV2 moduleScope;

  public WriteUsagesVisitor(Map<Tree, ScopeV2> scopesByRootTree) {
    super(scopesByRootTree);
  }

  @Override
  public void visitFileInput(FileInput tree) {
    createAndEnterScope(tree, null);
    moduleScope = currentScope();
    super.visitFileInput(tree);
  }

  @Override
  public void visitLambda(LambdaExpression lambdaExpression) {
    createAndEnterScope(lambdaExpression, currentScope());
    createParameters(lambdaExpression);
    super.visitLambda(lambdaExpression);
    leaveScope();
  }

  @Override
  public void visitDictCompExpression(DictCompExpression tree) {
    createAndEnterScope(tree, currentScope());
    super.visitDictCompExpression(tree);
    leaveScope();
  }

  /**
   * The scope of the decorator should be the parent scope of the function or class to which the decorator is assigned.
   * So we have to leave the function or class scope, visit the decorator and enter the previous scope.
   * See <a href="https://sonarsource.atlassian.net/browse/SONARPY-990">SONARPY-990</a>
   */
  @Override
  public void visitDecorator(Decorator tree) {
    leaveScope();
    super.visitDecorator(tree);
    enterScope(tree.parent());
  }

  @Override
  public void visitPyListOrSetCompExpression(ComprehensionExpression tree) {
    createAndEnterScope(tree, currentScope());
    super.visitPyListOrSetCompExpression(tree);
    leaveScope();
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    currentScope().addBindingUsage(functionDef.name(), UsageV2.Kind.FUNC_DECLARATION);
    createAndEnterScope(functionDef, currentScope());
    createTypeParameters(functionDef.typeParams());
    createParameters(functionDef);
    super.visitFunctionDef(functionDef);
    leaveScope();
  }

  private void createTypeParameters(@Nullable TypeParams typeParams) {
    Optional.ofNullable(typeParams)
      .map(TypeParams::typeParamsList)
      .stream()
      .flatMap(Collection::stream)
      .forEach(typeParam -> currentScope().addBindingUsage(typeParam.name(), UsageV2.Kind.TYPE_PARAM_DECLARATION));
  }


  private void createParameters(FunctionLike function) {
    ParameterList parameterList = function.parameters();
    if (parameterList == null || parameterList.all().isEmpty()) {
      return;
    }

    boolean hasSelf = false;
    if (function.isMethodDefinition()) {
      AnyParameter first = parameterList.all().get(0);
      if (first.is(Tree.Kind.PARAMETER)) {
        currentScope().createSelfParameter((Parameter) first);
        hasSelf = true;
      }
    }

    parameterList.nonTuple()
      .stream()
      .skip(hasSelf ? 1 : 0)
      .map(Parameter::name)
      .filter(Objects::nonNull)
      .forEach(param -> currentScope().addBindingUsage(param, UsageV2.Kind.PARAMETER));

    parameterList.all().stream()
      .filter(param -> param.is(Tree.Kind.TUPLE_PARAMETER))
      .map(TupleParameter.class::cast)
      .forEach(this::addTupleParamElementsToBindingUsage);
  }

  private void addTupleParamElementsToBindingUsage(TupleParameter param) {
    param.parameters().stream()
      .filter(p -> p.is(Tree.Kind.PARAMETER))
      .map(p -> ((Parameter) p).name())
      .forEach(name -> currentScope().addBindingUsage(name, UsageV2.Kind.PARAMETER));
    param.parameters().stream()
      .filter(p -> p.is(Tree.Kind.TUPLE_PARAMETER))
      .map(TupleParameter.class::cast)
      .forEach(this::addTupleParamElementsToBindingUsage);
  }

  @Override
  public void visitTypeAliasStatement(TypeAliasStatement typeAliasStatement) {
    currentScope().addBindingUsage(typeAliasStatement.name(), UsageV2.Kind.TYPE_ALIAS_DECLARATION);
    super.visitTypeAliasStatement(typeAliasStatement);
  }

  @Override
  public void visitClassDef(ClassDef classDef) {
    currentScope().addBindingUsage(classDef.name(), UsageV2.Kind.CLASS_DECLARATION);
    createAndEnterScope(classDef, currentScope());
    createTypeParameters(classDef.typeParams());
    super.visitClassDef(classDef);
    leaveScope();
  }

  @Override
  public void visitImportName(ImportName importName) {
    createImportedNames(importName.modules(), null, Collections.emptyList());
    super.visitImportName(importName);
  }

  @Override
  public void visitImportFrom(ImportFrom importFrom) {
    DottedName moduleTree = importFrom.module();
    String moduleName = moduleTree != null
      ? moduleTree.names().stream().map(Name::name).collect(Collectors.joining("."))
      : null;
    if (importFrom.isWildcardImport()) {
      // TODO: SONARPY-1781 handle wildcard import
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
      Name alias = module.alias();
      if (targetModuleName != null) {
        addBindingUsage(alias == null ? nameTree : alias, UsageV2.Kind.IMPORT);
      } else if (alias != null) {
        addBindingUsage(alias, UsageV2.Kind.IMPORT);
      } else if (dottedPrefix.isEmpty() && dottedNames.size() > 1) {
        // Submodule import
        addBindingUsage(nameTree, UsageV2.Kind.IMPORT);
      } else {
        // It's a simple case - no "from" imports or aliasing
        addBindingUsage(nameTree, UsageV2.Kind.IMPORT);
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
    addCompDeclarationParam(tree.loopExpression());
    super.visitComprehensionFor(tree);
  }

  private void addCompDeclarationParam(Tree tree) {
    boundNamesFromExpression(tree).forEach(name -> currentScope().addBindingUsage(name, UsageV2.Kind.COMP_DECLARATION));
  }

  private void createLoopVariables(ForStatement loopTree) {
    loopTree.expressions().forEach(expr ->
      boundNamesFromExpression(expr).forEach(name -> currentScope().addBindingUsage(name, UsageV2.Kind.LOOP_DECLARATION)));
  }

  @Override
  public void visitAssignmentStatement(AssignmentStatement pyAssignmentStatementTree) {
    SymbolUtils.assignmentsLhs(pyAssignmentStatementTree)
      .stream()
      .map(SymbolUtils::boundNamesFromExpression)
      .flatMap(Collection::stream)
      .forEach(name -> addBindingUsage(name, UsageV2.Kind.ASSIGNMENT_LHS));
    super.visitAssignmentStatement(pyAssignmentStatementTree);
  }

  @Override
  public void visitAnnotatedAssignment(AnnotatedAssignment annotatedAssignment) {
    if (annotatedAssignment.variable().is(Tree.Kind.NAME)) {
      Name variable = (Name) annotatedAssignment.variable();
      addBindingUsage(variable, UsageV2.Kind.ASSIGNMENT_LHS);
    }
    super.visitAnnotatedAssignment(annotatedAssignment);
  }

  @Override
  public void visitCompoundAssignment(CompoundAssignmentStatement pyCompoundAssignmentStatementTree) {
    if (pyCompoundAssignmentStatementTree.lhsExpression().is(Tree.Kind.NAME)) {
      addBindingUsage((Name) pyCompoundAssignmentStatementTree.lhsExpression(), UsageV2.Kind.COMPOUND_ASSIGNMENT_LHS);
    }
    super.visitCompoundAssignment(pyCompoundAssignmentStatementTree);
  }

  @Override
  public void visitAssignmentExpression(AssignmentExpression assignmentExpression) {
    addBindingUsage(assignmentExpression.lhsName(), UsageV2.Kind.ASSIGNMENT_LHS);
    super.visitAssignmentExpression(assignmentExpression);
  }

  @Override
  public void visitGlobalStatement(GlobalStatement globalStatement) {
    // Global statements are not binding usages, but we consider them as such for symbol creation
    globalStatement.variables().forEach(name -> {
      moduleScope.addBindingUsage(name, UsageV2.Kind.GLOBAL_DECLARATION);
      currentScope().addGlobalName(name);
    });
    super.visitGlobalStatement(globalStatement);
  }

  @Override
  public void visitNonlocalStatement(NonlocalStatement pyNonlocalStatementTree) {
    pyNonlocalStatementTree.variables().forEach(name -> currentScope().addNonLocalName(name));
    super.visitNonlocalStatement(pyNonlocalStatementTree);
  }

  @Override
  public void visitExceptClause(ExceptClause exceptClause) {
    boundNamesFromExpression(exceptClause.exceptionInstance()).forEach(name -> addBindingUsage(name, UsageV2.Kind.EXCEPTION_INSTANCE));
    super.visitExceptClause(exceptClause);
  }

  @Override
  public void visitWithItem(WithItem withItem) {
    boundNamesFromExpression(withItem.expression()).forEach(name -> addBindingUsage(name, UsageV2.Kind.WITH_INSTANCE));
    super.visitWithItem(withItem);
  }

  @Override
  public void visitCapturePattern(CapturePattern capturePattern) {
    addBindingUsage(capturePattern.name(), UsageV2.Kind.PATTERN_DECLARATION);
    super.visitCapturePattern(capturePattern);
  }

  private void addBindingUsage(Name nameTree, UsageV2.Kind usage) {
    currentScope().addBindingUsage(nameTree, usage);
  }
}
