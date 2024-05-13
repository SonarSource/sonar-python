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
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.python.semantic.v2.SymbolV2;

public class PropagationVisitor extends BaseTreeVisitor {
  private final Map<SymbolV2, Set<Assignment>> assignmentsByLhs;
  private final Map<Statement, Assignment> assignmentsByAssignmentStatement;

  public PropagationVisitor() {
    assignmentsByLhs = new HashMap<>();
    assignmentsByAssignmentStatement = new HashMap<>();
  }

  public Map<SymbolV2, Set<Assignment>> assignmentsByLhs() {
    return assignmentsByLhs;
  }

  public Map<Statement, Assignment> assignmentsByAssignmentStatement() {
    return assignmentsByAssignmentStatement;
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
      Assignment assignment = new Assignment(symbol, lhs, rhsExpression, assignmentsByLhs);
      assignmentsByAssignmentStatement.put(assignmentStatement, assignment);
      assignmentsByLhs.computeIfAbsent(symbol, s -> new HashSet<>()).add(assignment);
    }
  }

  public void processPropagations(Set<SymbolV2> trackedVars) {
    Set<Assignment> propagations = new HashSet<>();
    Set<SymbolV2> initializedVars = new HashSet<>();

    assignmentsByLhs.forEach((lhs, as) -> {
      if (trackedVars.contains(lhs)) {
        as.forEach(a -> a.computeDependencies(a.rhs(), trackedVars));
        propagations.addAll(as);
      }
    });

    applyPropagations(propagations, initializedVars, true);
    applyPropagations(propagations, initializedVars, false);
  }

  private static void applyPropagations(Set<Assignment> propagations, Set<SymbolV2> initializedVars, boolean checkDependenciesReadiness) {
    Set<Assignment> workSet = new HashSet<>(propagations);
    while (!workSet.isEmpty()) {
      Iterator<Assignment> iterator = workSet.iterator();
      Assignment propagation = iterator.next();
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
