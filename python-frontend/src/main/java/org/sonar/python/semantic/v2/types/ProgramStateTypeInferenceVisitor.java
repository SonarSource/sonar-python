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

import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

/**
 * Used in FlowSensitiveTypeInference to update name types based on program state
 */
public class ProgramStateTypeInferenceVisitor extends BaseTreeVisitor {
  private final TypeInferenceProgramState state;

  public ProgramStateTypeInferenceVisitor(TypeInferenceProgramState state) {
    this.state = state;
  }

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

  private static PythonType union(Stream<PythonType> types) {
    return types.reduce(UnionType::or).orElse(PythonType.UNKNOWN);
  }
}
