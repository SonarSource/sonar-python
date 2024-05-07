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
package org.sonar.python.semantic.v2.types;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.python.semantic.v2.SymbolV2;

public class PropagationVisitor extends BaseTreeVisitor {
  private final Map<SymbolV2, Set<Propagation>> propagationsByLhs;
  private final Map<Statement, Assignment> assignmentsByAssignmentStatement;
  private final Map<Statement, Definition> definitionsByDefinitionStatement;

  public PropagationVisitor() {
    propagationsByLhs = new HashMap<>();
    assignmentsByAssignmentStatement = new HashMap<>();
    definitionsByDefinitionStatement = new HashMap<>();
  }


  public Map<Statement, Assignment> assignmentsByAssignmentStatement() {
    return assignmentsByAssignmentStatement;
  }

  public Map<Statement, Definition> definitionsByDefinitionStatement() {
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
    Definition definition = new Definition(symbol, name);
    definitionsByDefinitionStatement.put(functionDef, definition);
    propagationsByLhs.computeIfAbsent(symbol, s -> new HashSet<>()).add(definition);
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

  private void processAssignment(Statement assignmentStatement, Expression lhsExpression, Expression rhsExpression){
    if (lhsExpression instanceof Name lhs && lhs.symbolV2() != null) {
      var symbol = lhs.symbolV2();
      Assignment assignment = new Assignment(symbol, lhs, rhsExpression, propagationsByLhs);
      assignmentsByAssignmentStatement.put(assignmentStatement, assignment);
      propagationsByLhs.computeIfAbsent(symbol, s -> new HashSet<>()).add(assignment);
    }
  }

  public void processPropagations(Set<SymbolV2> trackedVars) {
    Set<Propagation> propagations = new HashSet<>();
    Set<SymbolV2> initializedVars = new HashSet<>();

    propagationsByLhs.forEach((lhs, props) -> {
      if (trackedVars.contains(lhs)) {
        props.stream()
          .filter(Assignment.class::isInstance)
          .map(Assignment.class::cast)
          .forEach(a -> a.computeDependencies(a.rhs(), trackedVars));
        propagations.addAll(props);
      }
    });

    applyPropagations(propagations, initializedVars, true);
    applyPropagations(propagations, initializedVars, false);
  }

  private static void applyPropagations(Set<Propagation> propagations, Set<SymbolV2> initializedVars, boolean checkDependenciesReadiness) {
    Set<Propagation> workSet = new HashSet<>(propagations);
    while (!workSet.isEmpty()) {
      Iterator<Propagation> iterator = workSet.iterator();
      Propagation propagation = iterator.next();
      iterator.remove();
      if (!checkDependenciesReadiness || propagation.areDependenciesReady(initializedVars)) {
        boolean learnt = propagation.propagate(initializedVars);
        if (learnt) {
          workSet.addAll(propagation.dependents());
        }
      }
    }
  }
}
