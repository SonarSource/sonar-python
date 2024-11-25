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
package org.sonar.python.semantic.v2.types;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.tree.NameImpl;

public class PropagationVisitor extends BaseTreeVisitor {
  private final Map<SymbolV2, Set<Propagation>> propagationsByLhs;
  private final Map<Statement, Assignment> assignmentsByAssignmentStatement;
  private final Map<Statement, Set<Definition>> definitionsByDefinitionStatement;

  public PropagationVisitor() {
    propagationsByLhs = new HashMap<>();
    assignmentsByAssignmentStatement = new HashMap<>();
    definitionsByDefinitionStatement = new HashMap<>();
  }


  public Map<Statement, Assignment> assignmentsByAssignmentStatement() {
    return assignmentsByAssignmentStatement;
  }

  public Map<Statement, Set<Definition>> definitionsByDefinitionStatement() {
    return definitionsByDefinitionStatement;
  }

  public Map<SymbolV2, Set<Propagation>> propagationsByLhs() {
    return propagationsByLhs;
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    super.visitFunctionDef(functionDef);
    Name name = functionDef.name();
    var symbol = name.symbolV2();
    if (symbol == null) {
      return;
    }
    Definition definition = new Definition(symbol, name);
    definitionsByDefinitionStatement.computeIfAbsent(functionDef, k -> new HashSet<>()).add(definition);
    propagationsByLhs.computeIfAbsent(symbol, s -> new HashSet<>()).add(definition);
  }

  @Override
  public void visitClassDef(ClassDef classDef) {
    super.visitClassDef(classDef);
    var name = classDef.name();
    var symbol = name.symbolV2();
    if (symbol == null) {
      return;
    }
    var definition = new Definition(symbol, name);
    definitionsByDefinitionStatement.computeIfAbsent(classDef, k -> new HashSet<>()).add(definition);
    propagationsByLhs.computeIfAbsent(symbol, s -> new HashSet<>()).add(definition);
  }

  @Override
  public void visitParameter(Parameter parameter) {
    Optional.ofNullable(parameter.name())
      .ifPresent(name -> {
        var symbol = name.symbolV2();
        var parametedDefinition = new ParameterDefinition(symbol, name);
        propagationsByLhs.computeIfAbsent(symbol, s -> new HashSet<>()).add(parametedDefinition);
      });
  }

  @Override
  public void visitAssignmentStatement(AssignmentStatement assignmentStatement) {
    super.visitAssignmentStatement(assignmentStatement);
    if (assignmentStatement.lhsExpressions().stream().anyMatch(expressionList -> !expressionList.commas().isEmpty())) {
      return;
    }
    List<Expression> lhsExpressions = assignmentStatement.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .toList();
    if (lhsExpressions.size() != 1) {
      return;
    }
    processAssignment(assignmentStatement, lhsExpressions.get(0), assignmentStatement.assignedValue());
  }

  @Override
  public void visitCompoundAssignment(CompoundAssignmentStatement compoundAssignment) {
    super.visitCompoundAssignment(compoundAssignment);
    processAssignment(compoundAssignment, compoundAssignment.lhsExpression(), compoundAssignment.rhsExpression());
  }

  @Override
  public void visitAnnotatedAssignment(AnnotatedAssignment annotatedAssignment){
    super.visitAnnotatedAssignment(annotatedAssignment);
    Expression assignedValue = annotatedAssignment.assignedValue();
    if (assignedValue != null) {
      processAssignment(annotatedAssignment, annotatedAssignment.variable(), assignedValue);
    }
  }

  @Override
  public void visitImportName(ImportName importName) {
    super.visitImportName(importName);
    importName.modules()
      .forEach(aliasedName -> propagateImportToAliasedName(aliasedName, importName));
  }

  @Override
  public void visitImportFrom(ImportFrom importFrom) {
    super.visitImportFrom(importFrom);
    importFrom.importedNames()
      .forEach(aliasedName -> propagateImportToAliasedName(aliasedName, importFrom));
  }

  public void propagateImportToAliasedName(AliasedName aliasedName, Statement importName) {
    var alias = aliasedName.alias();
    List<Name> names = alias == null ? aliasedName.dottedName().names() : List.of(alias);
    for (Name name : names) {
      propagateToName(importName, name);
    }
  }

  private void propagateToName(Statement importName, Name name) {
    SymbolV2 symbolV2 = name.symbolV2();
    if (symbolV2 != null) {
      Definition definition = new Definition(symbolV2, name);
      definitionsByDefinitionStatement.computeIfAbsent(importName, k -> new HashSet<>()).add(definition);
      propagationsByLhs.computeIfAbsent(symbolV2, s -> new HashSet<>()).add(definition);
    }
  }

  @Override
  public void visitForStatement(ForStatement forStatement) {
    scan(forStatement.testExpressions());
    if (forStatement.testExpressions().size() == 1 && forStatement.expressions().size() == 1) {
      forStatement
        .testExpressions()
        .stream()
        .findFirst()
        .ifPresent(rhsExpression -> forStatement.expressions().stream()
            .findFirst()
            .filter(NameImpl.class::isInstance)
            .map(NameImpl.class::cast)
            .ifPresent(name -> {
              var symbol = name.symbolV2();
              var assignment = new LoopAssignment(symbol, name, rhsExpression);
              assignmentsByAssignmentStatement.put(forStatement, assignment);
              propagationsByLhs.computeIfAbsent(symbol, s -> new HashSet<>()).add(assignment);
            })
        );

    }
    scan(forStatement.body());
    scan(forStatement.elseClause());
  }

  private void processAssignment(Statement assignmentStatement, Expression lhsExpression, Expression rhsExpression){
    if (lhsExpression instanceof Name lhs) {
      var symbol = lhs.symbolV2();
      if (symbol == null) {
        return;
      }
      var assignment = new Assignment(symbol, lhs, rhsExpression);
      assignmentsByAssignmentStatement.put(assignmentStatement, assignment);
      propagationsByLhs.computeIfAbsent(symbol, s -> new HashSet<>()).add(assignment);
    }
  }


}
