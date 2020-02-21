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
  private final Set<Assignment> assignments = new HashSet<>();
  private final FunctionLike functionDef;

  public static void inferTypes(FunctionLike functionDef) {
    TypeInference visitor = new TypeInference(functionDef);
    functionDef.accept(visitor);
    visitor.processAssignments();
  }

  private TypeInference(FunctionLike functionDef) {
    this.functionDef = functionDef;
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
    if (symbol == null) {
      return;
    }

    Expression rhs = assignmentStatement.assignedValue();
    assignments.add(new Assignment(symbol, lhs, rhs));
  }

  private void processAssignments() {
    Set<Name> assignedNames = assignments.stream().map(a -> a.lhsName).collect(Collectors.toSet());
    for (Symbol variable : functionDef.localVariables()) {
      boolean hasMissingBindingUsage = variable.usages().stream()
        .filter(Usage::isBindingUsage)
        .anyMatch(u -> !assignedNames.contains(u.tree()));
      if (!hasMissingBindingUsage) {
        trackedVars.add((SymbolImpl) variable);
      }
    }

    Map<Symbol, Set<Assignment>> dependentAssignments = new HashMap<>();
    for (Assignment assignment : assignments) {
      if (!trackedVars.contains(assignment.lhs)) {
        continue;
      }
      Set<Symbol> rhsDependencies = dependencies(assignment.rhs);
      if (rhsDependencies.isEmpty()) {
        propagateType(assignment.lhs, assignment.rhs);
      } else {
        rhsDependencies.forEach(s -> dependentAssignments.computeIfAbsent(s, k -> new HashSet<>()).add(assignment));
      }
    }

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

  private Set<Symbol> dependencies(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      Symbol symbol = ((Name) expression).symbol();
      if (symbol != null && trackedVars.contains(symbol)) {
        return Collections.singleton(symbol);
      }
    }
    return Collections.emptySet();
  }

  private static class Assignment {
    private final SymbolImpl lhs;
    private final Name lhsName;
    private final Expression rhs;

    private Assignment(SymbolImpl lhs, Name lhsName, Expression rhs) {
      this.lhs = lhs;
      this.lhsName = lhsName;
      this.rhs = rhs;
    }
  }

}
