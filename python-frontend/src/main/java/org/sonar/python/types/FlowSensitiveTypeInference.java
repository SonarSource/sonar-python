/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.types;

import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.cfg.fixpoint.ForwardAnalysis;
import org.sonar.python.cfg.fixpoint.ProgramState;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.TypeInference.Assignment;
import org.sonar.python.types.TypeInference.MemberAccess;

import static org.sonar.plugins.python.api.tree.Tree.Kind.ANNOTATED_ASSIGNMENT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.ASSIGNMENT_STMT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.COMPOUND_ASSIGNMENT;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.REGULAR_ARGUMENT;

class FlowSensitiveTypeInference extends ForwardAnalysis {
  private final Set<Symbol> trackedVars;
  private final Map<QualifiedExpression, MemberAccess> memberAccessesByQualifiedExpr;
  private final Map<Statement, Assignment> assignmentsByAssignmentStatement;
  private final Map<String, InferredType> parameterTypesByName;

  public FlowSensitiveTypeInference(Set<Symbol> trackedVars, Map<QualifiedExpression, MemberAccess> memberAccessesByQualifiedExpr,
                                    Map<Statement, Assignment> assignmentsByAssignmentStatement, Map<String, InferredType> parameterTypesByName) {
    this.trackedVars = trackedVars;
    this.memberAccessesByQualifiedExpr = memberAccessesByQualifiedExpr;
    this.assignmentsByAssignmentStatement = assignmentsByAssignmentStatement;
    this.parameterTypesByName = parameterTypesByName;
  }

  @Override
  public ProgramState initialState() {
    TypeInferenceProgramState initialState = new TypeInferenceProgramState();
    for (Symbol variable : trackedVars) {
      InferredType inferredType = parameterTypesByName.get(variable.name());
      initialState.setTypes(variable, inferredType != null ? Collections.singleton(inferredType) : Collections.emptySet());
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
    } else if (element.is(COMPOUND_ASSIGNMENT)) {
      // Assumption: compound assignments don't change types
      updateTree(element, state);
    } else if (element.is(ANNOTATED_ASSIGNMENT)) {
      AnnotatedAssignment assignment = (AnnotatedAssignment) element;
      Expression assignedValue = assignment.assignedValue();
      if (assignedValue != null) {
        // update rhs
        updateTree(assignedValue, state);
        handleAssignment(assignment, state);
        // update lhs
        updateTree(assignment.variable(), state);
      }
    } else {
      element.accept(new IsInstanceVisitor(state));
      updateTree(element, state);
    }
  }

  private static class IsInstanceVisitor extends BaseTreeVisitor {
    private final TypeInferenceProgramState state;

    public IsInstanceVisitor(TypeInferenceProgramState state) {
      this.state = state;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && "isinstance".equals(calleeSymbol.fullyQualifiedName()) && callExpression.arguments().size() == 2) {
        Symbol firstArgumentSymbol = getFirstArgumentSymbol(callExpression);
        if (firstArgumentSymbol != null) {
          state.setTypes(firstArgumentSymbol, Collections.singleton(InferredTypes.anyType()));
        }
      }
      super.visitCallExpression(callExpression);
    }

    @CheckForNull
    private Symbol getFirstArgumentSymbol(CallExpression callExpression) {
      Argument argument = callExpression.arguments().get(0);
      if (argument.is(REGULAR_ARGUMENT) && ((RegularArgument) argument).expression().is(NAME)) {
        Name variableName = (Name) ((RegularArgument) argument).expression();
        if (state.getTypes(variableName.symbol()).stream().anyMatch(InferredTypes::containsDeclaredType)) {
          return variableName.symbol();
        }
      }
      return null;
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

  private void handleAssignment(Statement assignmentStatement, TypeInferenceProgramState programState) {
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
