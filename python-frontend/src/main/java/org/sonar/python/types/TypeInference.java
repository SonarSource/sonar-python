/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.types;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionLike;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.SymbolImpl;

public class TypeInference extends BaseTreeVisitor {

  private final Set<SymbolImpl> trackedVars = new HashSet<>();
  private final Set<Symbol> initializedVars = new HashSet<>();
  private final Map<Symbol, Set<Assignment>> dependentAssignments = new HashMap<>();
  private final Set<Name> visitedLhs = new HashSet<>();

  public static void inferTypes(FunctionLike functionDef) {
    TypeInference visitor = new TypeInference(functionDef);
    functionDef.accept(visitor);
    visitor.processDependencies();
    visitor.resetTypeOfSymbolWhenMissingAssignment();
  }

  private TypeInference(FunctionLike functionDef) {
    for (Symbol variable : functionDef.localVariables()) {
      Set<Usage.Kind> unsupportedKinds = variable.usages().stream()
        .map(Usage::kind)
        .filter(k -> k != Usage.Kind.ASSIGNMENT_LHS && k != Usage.Kind.OTHER)
        .collect(Collectors.toSet());
      if (unsupportedKinds.isEmpty()) {
        trackedVars.add((SymbolImpl) variable);
      }
    }
  }

  private void resetTypeOfSymbolWhenMissingAssignment() {
    for (SymbolImpl var : trackedVars) {
      Optional<Usage> missedBindingUsage = var.usages().stream()
        .filter(Usage::isBindingUsage)
        .filter(u -> !visitedLhs.contains(u.tree()))
        .findAny();
      if (missedBindingUsage.isPresent()) {
        var.setInferredType(InferredTypes.anyType());
      }
    }
  }

  @Override
  public void visitAssignmentStatement(AssignmentStatement assignmentStatement) {
    super.visitAssignmentStatement(assignmentStatement);
    List<Expression> lhsExpressions = assignmentStatement.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .collect(Collectors.toList());
    if (lhsExpressions.size() != 1) {
      return;
    }
    Expression lhsExpression = lhsExpressions.get(0);
    if (!lhsExpression.is(Tree.Kind.NAME)) {
      return;
    }
    Name lhs = (Name) lhsExpression;
    SymbolImpl symbol = (SymbolImpl) lhs.symbol();
    if (symbol == null || !trackedVars.contains(symbol)) {
      return;
    }
    visitedLhs.add(lhs);

    Expression rhs = assignmentStatement.assignedValue();
    Set<Symbol> rhsDependencies = dependencies(rhs);
    if (rhsDependencies.isEmpty()) {
      propagateType(symbol, rhs);
    } else {
      Assignment assignment = new Assignment(symbol, rhs);
      rhsDependencies.forEach(s -> dependentAssignments.computeIfAbsent(s, k -> new HashSet<>()).add(assignment));
    }
  }

  private void processDependencies() {
    Set<Assignment> workSet = new HashSet<>();
    for (Set<Assignment> dependent : dependentAssignments.values()) {
      workSet.addAll(dependent);
    }
    while (!workSet.isEmpty()) {
      Iterator<Assignment> iterator = workSet.iterator();
      Assignment assignment = iterator.next();
      iterator.remove();
      if (!initializedVars.containsAll(dependencies(assignment.rhs))) {
        continue;
      }
      boolean learnt = propagateType(assignment.lhs, assignment.rhs);
      if (learnt) {
        workSet.addAll(dependentAssignments.getOrDefault(assignment.lhs, Collections.emptySet()));
      }
    }
  }

  /** @return true if the propagation effectively changed the inferred type of lhs */
  private boolean propagateType(SymbolImpl lhs, Expression rhs) {
    InferredType rhsType = rhs.type();
    if (initializedVars.add(lhs)) {
      lhs.setInferredType(rhsType);
      return true;
    } else {
      InferredType currentType = lhs.inferredType();
      InferredType newType = InferredTypes.or(rhsType, currentType);
      lhs.setInferredType(newType);
      return !newType.equals(currentType);
    }
  }

  private static Set<Symbol> dependencies(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      Symbol symbol = ((Name) expression).symbol();
      if (symbol != null) {
        return Collections.singleton(symbol);
      }
    }
    return Collections.emptySet();
  }

  private static class Assignment {
    private final SymbolImpl lhs;
    private final Expression rhs;

    private Assignment(SymbolImpl lhs, Expression rhs) {
      this.lhs = lhs;
      this.rhs = rhs;
    }
  }

}
