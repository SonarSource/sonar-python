/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic.v2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
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
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
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
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TupleParameter;
import org.sonar.plugins.python.api.tree.TypeAliasStatement;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.tree.TypeParams;
import org.sonar.plugins.python.api.tree.WithItem;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.DictCompExpressionImpl;

import static org.sonar.python.semantic.SymbolUtils.boundNamesFromExpression;

public class SymbolTableBuilderV2 extends BaseTreeVisitor {
  //private final PythonFile pythonFile;
  private final ProjectLevelSymbolTable projectLevelSymbolTable;
  private String fullyQualifiedModuleName;
  private List<String> filePath;
  private FileInput fileInput;
  private ScopeV2 scope;
  private Map<Tree, ScopeV2> scopesByRootTree;

  public SymbolTableBuilderV2(ProjectLevelSymbolTable projectLevelSymbolTable) {
    //this.pythonFile = pythonFile;
    this.projectLevelSymbolTable = projectLevelSymbolTable;
    fullyQualifiedModuleName = null;
    filePath = null;
    fileInput = null;
  }

  public SymbolTableBuilderV2() {
    this.projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    fullyQualifiedModuleName = null;
    filePath = null;
    fileInput = null;
  }

  public SymbolTableBuilderV2(String packageName, PythonFile pythonFile) {
    this(packageName, pythonFile, ProjectLevelSymbolTable.empty());
  }

  public SymbolTableBuilderV2(String packageName, PythonFile pythonFile, ProjectLevelSymbolTable projectLevelSymbolTable) {
    //this.pythonFile = pythonFile;
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
    fileInput.accept(new SymbolTableBuilderV2.FirstPhaseVisitor());
    fileInput.accept(new SymbolTableBuilderV2.SecondPhaseVisitor());
  }

  private class FirstPhaseVisitor extends ScopeVisitor {
    @Override
    public void visitFileInput(FileInput tree) {
      createScope(tree, null);
      enterScope(tree);
      moduleScope = currentScope();
      //TODO: Should TypeShed symbols be resolved here?
/*      Map<String, Symbol> typeShedSymbols = TypeShed.builtinSymbols();
      for (String name : BuiltinSymbols.all()) {
        currentScope().createBuiltinSymbol(name, typeShedSymbols);
      }*/
      super.visitFileInput(tree);
    }

    @Override
    public void visitLambda(LambdaExpression lambdaExpression) {
      createScope(lambdaExpression, currentScope());
      enterScope(lambdaExpression);
      createParameters(lambdaExpression);
      super.visitLambda(lambdaExpression);
      leaveScope();
    }

    @Override
    public void visitDictCompExpression(DictCompExpressionImpl tree) {
      createScope(tree, currentScope());
      enterScope(tree);
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
      createScope(tree, currentScope());
      enterScope(tree);
      super.visitPyListOrSetCompExpression(tree);
      leaveScope();
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      String functionName = functionDef.name().name();
      //String fullyQualifiedName = getFullyQualifiedName(functionName);
      //currentScope().addFunctionSymbol(pyFunctionDefTree, fullyQualifiedName);
      currentScope().addBindingUsage(functionDef.name(), UsageV2.Kind.FUNC_DECLARATION, null);
      createScope(functionDef, currentScope());
      enterScope(functionDef);
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
        .forEach(typeParam -> currentScope().addBindingUsage(typeParam.name(), UsageV2.Kind.TYPE_PARAM_DECLARATION, null));
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
        .forEach(param -> currentScope().addBindingUsage(param, UsageV2.Kind.PARAMETER, null));

      parameterList.all().stream()
        .filter(param -> param.is(Tree.Kind.TUPLE_PARAMETER))
        .map(TupleParameter.class::cast)
        .forEach(this::addTupleParamElementsToBindingUsage);
    }

    private void addTupleParamElementsToBindingUsage(TupleParameter param) {
      param.parameters().stream()
        .filter(p -> p.is(Tree.Kind.PARAMETER))
        .map(p -> ((Parameter) p).name())
        .forEach(name -> currentScope().addBindingUsage(name, UsageV2.Kind.PARAMETER, null));
      param.parameters().stream()
        .filter(p -> p.is(Tree.Kind.TUPLE_PARAMETER))
        .map(TupleParameter.class::cast)
        .forEach(this::addTupleParamElementsToBindingUsage);
    }

    @Override
    public void visitTypeAliasStatement(TypeAliasStatement typeAliasStatement) {
      currentScope().addBindingUsage(typeAliasStatement.name(), UsageV2.Kind.TYPE_ALIAS_DECLARATION, null);
      super.visitTypeAliasStatement(typeAliasStatement);
    }

    @Override
    public void visitClassDef(ClassDef classDef) {
      currentScope().addBindingUsage(classDef.name(), UsageV2.Kind.CLASS_DECLARATION, null);
      createScope(classDef, currentScope());
      enterScope(classDef);
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
        // FIXME: handle wildcard import
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
          addBindingUsage(alias == null ? nameTree : alias, UsageV2.Kind.IMPORT, null);
        } else if (alias != null) {
          addBindingUsage(alias, UsageV2.Kind.IMPORT, null);
        } else if (dottedPrefix.isEmpty() && dottedNames.size() > 1) {
          // Submodule import
          addBindingUsage(nameTree, UsageV2.Kind.IMPORT, null);
        } else {
          // It's a simple case - no "from" imports or aliasing
          addBindingUsage(nameTree, UsageV2.Kind.IMPORT, null);
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
      boundNamesFromExpression(tree).forEach(name -> currentScope().addBindingUsage(name, UsageV2.Kind.COMP_DECLARATION, null));
    }

    private void createLoopVariables(ForStatement loopTree) {
      loopTree.expressions().forEach(expr ->
        boundNamesFromExpression(expr).forEach(name -> currentScope().addBindingUsage(name, UsageV2.Kind.LOOP_DECLARATION, null)));
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement pyAssignmentStatementTree) {
      List<Expression> lhs = SymbolUtils.assignmentsLhs(pyAssignmentStatementTree);

      //assignmentLeftHandSides.addAll(lhs);

      lhs.forEach(expression -> boundNamesFromExpression(expression).forEach(name -> addBindingUsage(name, UsageV2.Kind.ASSIGNMENT_LHS)));

      super.visitAssignmentStatement(pyAssignmentStatementTree);
    }

    @Override
    public void visitAnnotatedAssignment(AnnotatedAssignment annotatedAssignment) {
      if (annotatedAssignment.variable().is(Tree.Kind.NAME)) {
        Name variable = (Name) annotatedAssignment.variable();
        addBindingUsage(variable, UsageV2.Kind.ASSIGNMENT_LHS, null);
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
      globalStatement.variables().forEach(name -> moduleScope.addBindingUsage(name, UsageV2.Kind.GLOBAL_DECLARATION, null));
      super.visitGlobalStatement(globalStatement);
    }

    @Override
    public void visitNonlocalStatement(NonlocalStatement nonlocalStatement) {
      // FIXME: handle nonlocal statements
      super.visitNonlocalStatement(nonlocalStatement);
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
      addBindingUsage(nameTree, usage, null);
    }

    private void addBindingUsage(Name nameTree, UsageV2.Kind usage, @Nullable String fullyQualifiedName) {
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
    public void visitDictCompExpression(DictCompExpressionImpl tree) {
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
      if (tree.is(Tree.Kind.PARAMETER_TYPE_ANNOTATION) || tree.is(Tree.Kind.RETURN_TYPE_ANNOTATION)) {
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
    public void visitClassDef(ClassDef classDef) {
      scan(classDef.args());
      scan(classDef.decorators());
      enterScope(classDef);
      scan(classDef.name());
//      resolveTypeHierarchy(classDef, classDef.name().symbol(), pythonFile, scopesByRootTree.get(fileInput).symbolsByName);
      scan(classDef.body());
      leaveScope();
    }

    @Override
    public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
      // We need to firstly create symbol for qualifier
      super.visitQualifiedExpression(qualifiedExpression);
//      if (qualifiedExpression.qualifier() instanceof Name name) {
//        var qualifierSymbol = name.symbolV2();
//        if (qualifierSymbol != null) {
//          UsageV2.Kind usageKind = assignmentLeftHandSides.contains(qualifiedExpression) ? UsageV2.Kind.ASSIGNMENT_LHS : UsageV2.Kind.OTHER;
//          qualifierSymbol.addUsage(qualifiedExpression.name(), usageKind);
//        }
//      }
    }

    @Override
    public void visitName(Name name) {
      if (!name.isVariable()) {
        return;
      }
      addSymbolUsage(name);
      super.visitName(name);
    }

    private void addSymbolUsage(Name name) {
      var scope = scopesByRootTree.get(currentScopeRootTree());
      var symbol = scope.resolve(name.name());
      if (symbol != null && symbol.usages().stream().noneMatch(usage -> usage.tree().equals(name))) {
        symbol.addUsage(name, UsageV2.Kind.OTHER);
      }
    }
  }

  private void createScope(Tree tree, @Nullable ScopeV2 parent) {
    scopesByRootTree.put(tree, new ScopeV2(parent, tree));
  }

  private class ScopeVisitor extends BaseTreeVisitor {

    private Deque<Tree> scopeRootTrees = new LinkedList<>();
    protected ScopeV2 moduleScope;

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
  }
}
