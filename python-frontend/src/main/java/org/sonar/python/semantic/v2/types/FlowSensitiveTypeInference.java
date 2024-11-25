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

import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
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
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ForwardAnalysis;
import org.sonar.python.cfg.fixpoint.ProgramState;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.TypeTable;
import org.sonar.python.types.v2.PythonType;

public class FlowSensitiveTypeInference extends ForwardAnalysis {
  private final Set<SymbolV2> trackedVars;
  private final Map<Statement, Assignment> assignmentsByAssignmentStatement;
  private final Map<Statement, Set<Definition>> definitionsByDefinitionStatement;
  private final Map<String, PythonType> parameterTypesByName;

  private final TypeTable typeTable;
  private final IsInstanceVisitor isInstanceVisitor;


  public FlowSensitiveTypeInference(
    TypeTable typeTable, Set<SymbolV2> trackedVars,
    Map<Statement, Assignment> assignmentsByAssignmentStatement,
    Map<Statement, Set<Definition>> definitionsByDefinitionStatement,
    Map<String, PythonType> parameterTypesByName
  ) {
    this.trackedVars = trackedVars;
    this.assignmentsByAssignmentStatement = assignmentsByAssignmentStatement;
    this.definitionsByDefinitionStatement = definitionsByDefinitionStatement;
    this.parameterTypesByName = parameterTypesByName;

    this.typeTable = typeTable;
    this.isInstanceVisitor = new IsInstanceVisitor(typeTable);
  }

  @Override
  public ProgramState initialState() {
    TypeInferenceProgramState initialState = new TypeInferenceProgramState();
    for (SymbolV2 variable : trackedVars) {
      initialState.setTypes(variable, Set.of());
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
    } else if (element instanceof FunctionDef functionDef) {
      handleDefinitions(functionDef, state);
    } else if (element instanceof ClassDef classDef) {
      handleDefinitions(classDef, state);
    } else if (element instanceof ImportName importName) {
      handleDefinitions(importName, state);
    } else if (element instanceof ImportFrom importFrom) {
      handleDefinitions(importFrom, state);
    } else if (element instanceof Parameter parameter) {
      handleParameter(parameter, state);
    } else if (isForLoopAssignment(element)) {
      handleLoopAssignment(element, state);
    } else {
      isInstanceVisitor.setState(state);
      element.accept(isInstanceVisitor);
      updateTree(element, state);
    }
  }

  @Override
  public TypeInferenceProgramState compute(ControlFlowGraph cfg) {
    return (TypeInferenceProgramState) super.compute(cfg);
  }

  private void handleParameter(Parameter parameter, TypeInferenceProgramState state) {
    var name = parameter.name();

    if (name == null || !trackedVars.contains(name.symbolV2())) {
      return;
    }
    SymbolV2 symbol = name.symbolV2();
    if (symbol == null) {
      return;
    }

    var type = parameterTypesByName.getOrDefault(name.name(), PythonType.UNKNOWN);
    state.setTypes(symbol, new HashSet<>(Set.of(type)));
    updateTree(name, state);
  }

  private void updateTree(Tree tree, TypeInferenceProgramState state) {
    tree.accept(new ProgramStateTypeInferenceVisitor(state, typeTable));
  }


  private static boolean isForLoopAssignment(Tree tree) {
    return tree instanceof Name && tree.parent() instanceof ForStatement forStatement && forStatement.expressions().contains(tree);
  }

  private void handleLoopAssignment(Tree element, TypeInferenceProgramState state) {
    Optional.of(element)
      .map(Tree::parent)
      .filter(ForStatement.class::isInstance)
      .map(ForStatement.class::cast)
      .ifPresent(forStatement -> {
        forStatement.testExpressions().forEach(t -> updateTree(t, state));
        Optional.ofNullable(assignmentsByAssignmentStatement.get(forStatement))
          .filter(assignment -> trackedVars.contains(assignment.lhsSymbol()))
          .ifPresent(assignment -> Optional.of(assignment)
            .map(Assignment::rhsType)
            .ifPresent(collectionItemType -> state.setTypes(assignment.lhsSymbol(), Set.of(collectionItemType)))
          );
      });
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

  private void handleDefinitions(Statement definitionStatement, TypeInferenceProgramState programState) {
    Optional.ofNullable(definitionsByDefinitionStatement.get(definitionStatement))
      .ifPresent(definitions -> definitions.forEach(d -> {
        SymbolV2 symbol = d.lhsSymbol();
        if (trackedVars.contains(symbol)) {
          programState.setTypes(symbol, Set.of(d.lhsName().typeV2()));
        }
      }));
  }
}
