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
package org.sonar.python.semantic.v2.types;

import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.python.semantic.v2.TypeTable;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeUtils;

/**
 * Used in FlowSensitiveTypeInference to update name types based on program state
 */
public class ProgramStateTypeInferenceVisitor extends TrivialTypePropagationVisitor {
  private final TypeInferenceProgramState state;

  public ProgramStateTypeInferenceVisitor(TypeInferenceProgramState state, TypeTable typeTable) {
    super(typeTable);
    this.state = state;
  }

  @Override
  public void visitName(Name name) {
    Optional.ofNullable(name.symbolV2()).ifPresent(symbol -> {
      Set<PythonType> pythonTypes = state.getTypes(symbol);
      if (!pythonTypes.isEmpty()) {
        ((NameImpl) name).typeV2(union(pythonTypes));
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
    scan(qualifiedExpression.qualifier());
    if (qualifiedExpression.name() instanceof NameImpl name) {
      Optional.of(qualifiedExpression.qualifier())
        .map(Expression::typeV2)
        .flatMap(t -> t.resolveMember(name.name()))
        .ifPresent(name::typeV2);
    }
  }

  private static PythonType union(Set<PythonType> types) {
    return types.stream().collect(TypeUtils.toUnionType());
  }
}
