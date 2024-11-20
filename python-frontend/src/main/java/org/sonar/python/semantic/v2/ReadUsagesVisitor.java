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

import java.util.Map;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;

/**
 * Read (i.e. non-binding) usages have to be visited in a second phase.
 * They can't be visited in the same phase as write (i.e. binding) usages,
 * since a read usage may appear in the syntax tree "before" it's declared (written).
 */
public class ReadUsagesVisitor extends ScopeVisitor {
  public ReadUsagesVisitor(Map<Tree, ScopeV2> scopesByRootTree) {
    super(scopesByRootTree);
  }

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
    scan(classDef.body());
    leaveScope();
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
    var scope = currentScope();
    var symbol = scope.resolve(name.name());
    if (symbol != null && symbol.usages().stream().noneMatch(usage -> usage.tree().equals(name))) {
      symbol.addUsage(name, UsageV2.Kind.OTHER);
    }
  }
}
