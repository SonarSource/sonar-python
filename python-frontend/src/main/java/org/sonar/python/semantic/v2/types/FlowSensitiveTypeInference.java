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

import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ForwardAnalysis;
import org.sonar.python.cfg.fixpoint.ProgramState;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

public class FlowSensitiveTypeInference extends ForwardAnalysis {
  private final Set<SymbolV2> trackedVars;
  private final Map<Statement, Assignment> assignmentsByAssignmentStatement;
  private final Map<String, PythonType> parameterTypesByName;

  public FlowSensitiveTypeInference(Set<SymbolV2> trackedVars,
                                    Map<Statement, Assignment> assignmentsByAssignmentStatement, Map<String, PythonType> parameterTypesByName) {
    this.trackedVars = trackedVars;
    this.assignmentsByAssignmentStatement = assignmentsByAssignmentStatement;
    this.parameterTypesByName = parameterTypesByName;
  }

  @Override
  public ProgramState initialState() {
    TypeInferenceProgramState initialState = new TypeInferenceProgramState();
    for (SymbolV2 variable : trackedVars) {
      var pythonTypeSet = Optional.of(variable.name())
        .map(parameterTypesByName::get)
        .map(Set::of)
        .orElseGet(Set::of);
      initialState.setTypes(variable, pythonTypeSet);
    }
    return initialState;
  }

  @Override
  public void updateProgramState(Tree element, ProgramState programState) {
    TypeInferenceProgramState state = (TypeInferenceProgramState) programState;
    if (element instanceof AssignmentStatement assignment) {
      // update rhs
      updateTree(assignment.assignedValue(), state);
      handleAssignment(assignment, state);
      // update lhs
      assignment.lhsExpressions().forEach(lhs -> updateTree(lhs, state));
    } else if (element instanceof CompoundAssignmentStatement) {
      // Assumption: compound assignments don't change types
      updateTree(element, state);
    } else if (element instanceof AnnotatedAssignment assignment) {
      var assignedValue = assignment.assignedValue();
      if (assignedValue != null) {
        handleAssignment(assignment, state);
        updateTree(assignedValue, state);
        // update lhs
        updateTree(assignment.variable(), state);
      }
    } else {
      // TODO: isinstance visitor
      updateTree(element, state);
    }
  }

  private static void updateTree(Tree tree, TypeInferenceProgramState state) {
    tree.accept(new BaseTreeVisitor() {
      @Override
      public void visitName(Name name) {
        Optional.ofNullable(name.symbolV2()).ifPresent(symbol -> {
          Set<PythonType> pythonTypes = state.getTypes(symbol);
          if (!pythonTypes.isEmpty()) {
            ((NameImpl) name).typeV2(union(pythonTypes.stream()));
          }
        });
        super.visitName(name);
      }

      @Override
      public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
        // skip inner functions
      }
    });
  }

  public static PythonType or(PythonType t1, PythonType t2) {
    return UnionType.or(t1, t2);
  }

  public static PythonType union(Stream<PythonType> types) {
    return types.reduce(FlowSensitiveTypeInference::or).orElse(PythonType.UNKNOWN);
  }

  private void handleAssignment(Statement assignmentStatement, TypeInferenceProgramState programState) {
    Optional.ofNullable(assignmentsByAssignmentStatement.get(assignmentStatement))
      .ifPresent(assignment -> {
        if (trackedVars.contains(assignment.lhsSymbol())) {
          Expression rhs = assignment.rhs();
          // strong update
          if (rhs instanceof Name rhsName && trackedVars.contains(rhsName.symbolV2())) {
            SymbolV2 rhsSymbol = rhsName.symbolV2();
            programState.setTypes(assignment.lhsSymbol(), programState.getTypes(rhsSymbol));
          } else {
            programState.setTypes(assignment.lhsSymbol(), Set.of(rhs.typeV2()));
          }
        }
      });
  }
}
