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

import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.types.HasTypeDependencies;

public class TypeDependenciesCalculator {
  static final Set<Tree.Kind> SAME_TYPE_PRODUCING_BINARY_EXPRESSION_KINDS = EnumSet.of(
    Tree.Kind.PLUS,
    Tree.Kind.MINUS,
    Tree.Kind.MULTIPLICATION,
    Tree.Kind.DIVISION,
    Tree.Kind.FLOOR_DIVISION,
    Tree.Kind.MODULO,
    Tree.Kind.POWER
  );

  public boolean hasTypeDependencies(Expression expression) {
    return expression instanceof HasTypeDependencies;
  }

  public List<Expression> getTypeDependencies(Expression expression) {
    if (expression instanceof BinaryExpression binaryExpression) {
      return calculateBinaryExpressionTypeDependencies(binaryExpression);
    } else if (expression instanceof HasTypeDependencies hasTypeDependencies) {
      // SONARPY-2417 Once we get rid of v1 type inference -
      // we won’t need a HasTypeDependencies interface implemented by a tree model classes.
      // The implementation of the logic is still needed for v2 type inference
      // but to keep tree model decoupled from the type inference and type model -
      // the logic should be moved here as it is done for BinaryExpression
      return hasTypeDependencies.typeDependencies();
    }
    return List.of();
  }

  private static List<Expression> calculateBinaryExpressionTypeDependencies(BinaryExpression binaryExpression) {
    if (SAME_TYPE_PRODUCING_BINARY_EXPRESSION_KINDS.contains(binaryExpression.getKind())
        || binaryExpression.is(Tree.Kind.AND, Tree.Kind.OR)) {
      return List.of(binaryExpression.leftOperand(), binaryExpression.rightOperand());
    }
    return List.of();
  }


}
