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
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.cfg.fixpoint.ForwardAnalysis;
import org.sonar.python.cfg.fixpoint.ProgramState;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.TypeInference.Assignment;
import org.sonar.python.types.TypeInference.MemberAccess;

import static org.sonar.plugins.python.api.tree.Tree.Kind.ASSIGNMENT_STMT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;

class FlowSensitiveTypeInference extends ForwardAnalysis {
  private final Set<Symbol> trackedVars;
  private final Map<QualifiedExpression, MemberAccess> memberAccessesByQualifiedExpr;
  private final Map<AssignmentStatement, Assignment> assignmentsByAssignmentStatement;

  public FlowSensitiveTypeInference(Set<Symbol> trackedVars, Map<QualifiedExpression, MemberAccess> memberAccessesByQualifiedExpr,
                                    Map<AssignmentStatement, Assignment> assignmentsByAssignmentStatement) {
    this.trackedVars = trackedVars;
    this.memberAccessesByQualifiedExpr = memberAccessesByQualifiedExpr;
    this.assignmentsByAssignmentStatement = assignmentsByAssignmentStatement;
  }

  @Override
  public ProgramState initialState() {
    TypeInferenceProgramState initialState = new TypeInferenceProgramState();
    for (Symbol variable : trackedVars) {
      initialState.setTypes(variable, Collections.emptySet());
    }
    return initialState;
  }

  @Override
  public void updateProgramState(Tree element, ProgramState programState) {
    TypeInferenceProgramState state = (TypeInferenceProgramState) programState;
    if (element.is(ASSIGNMENT_STMT)) {
      AssignmentStatement assignment = (AssignmentStatement) element;
      // update rhs
      updateTree(assignment.assignedValue(), state);
      handleAssignment(assignment, state);
      // update lhs
      assignment.lhsExpressions().forEach(lhs -> updateTree(lhs, state));
    } else {
      updateTree(element, state);
    }
  }

  private void updateTree(Tree tree, TypeInferenceProgramState state) {
    tree.accept(new BaseTreeVisitor() {
      @Override
      public void visitName(Name name) {
        Optional.ofNullable(name.symbol()).ifPresent(symbol -> {
          Set<InferredType> inferredTypes = state.getTypes(symbol);
          if (!inferredTypes.isEmpty()) {
            ((NameImpl) name).setInferredType(InferredTypes.union(inferredTypes.stream()));
          }
        });
        super.visitName(name);
      }

      @Override
      public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
        // skip inner functions
      }

      @Override
      public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
        super.visitQualifiedExpression(qualifiedExpression);
        Optional.ofNullable(memberAccessesByQualifiedExpr.get(qualifiedExpression))
          .ifPresent(memberAccess -> memberAccess.propagate(Collections.emptySet()));
      }
    });
  }

  private void handleAssignment(AssignmentStatement assignmentStatement, TypeInferenceProgramState programState) {
    Optional.ofNullable(assignmentsByAssignmentStatement.get(assignmentStatement)).ifPresent(assignment -> {
      if (trackedVars.contains(assignment.lhs)) {
        Expression rhs = assignment.rhs;
        // strong update
        if (rhs.is(NAME) && trackedVars.contains(((Name) rhs).symbol())) {
          Symbol rhsSymbol = ((Name) rhs).symbol();
          programState.setTypes(assignment.lhs, programState.getTypes(rhsSymbol));
        } else {
          programState.setTypes(assignment.lhs, Collections.singleton(rhs.type()));
        }
      }
    });
  }
}
