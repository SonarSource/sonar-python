/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.semantic.v2.types.typecalculator;

import java.util.Optional;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.semantic.v2.types.TypeInferenceMatcher;
import org.sonar.python.semantic.v2.types.TypeInferenceMatchers;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

public class QualifiedExpressionCalculator {
  private static final TypeInferenceMatcher IS_PROPERTY_TYPE = TypeInferenceMatcher.of(
    TypeInferenceMatchers.isSubtypeOf("property"));

  private final TypePredicateContext typePredicateContext;

  public QualifiedExpressionCalculator(TypePredicateContext typePredicateContext) {
    this.typePredicateContext = typePredicateContext;
  }

  public PythonType calculate(QualifiedExpression qualifiedExpression) {
    Name name = qualifiedExpression.name();
    return Optional.of(qualifiedExpression.qualifier())
      .map(Expression::typeV2)
      .flatMap(t -> t.resolveMember(name.name()))
      .map(this::handlePropertyFunction)
      .orElse(PythonType.UNKNOWN);
  }

  private PythonType handlePropertyFunction(PythonType type) {
    // If a member access is a method with a "property" annotation, we consider the resulting type to be the return type of the method
    if (type instanceof FunctionType functionType && hasFunctionPropertyDecorator(functionType)) {
      return functionType.returnType();
    }
    return type;
  }

  private boolean hasFunctionPropertyDecorator(FunctionType functionType) {
    return functionType.decorators().stream().anyMatch(t -> isProperty(t.type()));
  }

  private boolean isProperty(PythonType type) {
    return IS_PROPERTY_TYPE.evaluate(type, typePredicateContext).isTrue();
  }
}
