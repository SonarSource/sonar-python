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
package org.sonar.python.tree;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.HasTypeDependencies;
import org.sonar.python.types.InferredTypes;

public class ConditionalExpressionImpl extends PyTree implements ConditionalExpression, HasTypeDependencies {
  private final Expression trueExpression;
  private final Token ifToken;
  private final Expression condition;
  private final Token elseToken;
  private final Expression falseExpression;

  public ConditionalExpressionImpl(Expression trueExpression,
                                   Token ifToken, Expression condition, Token elseToken, Expression falseExpression) {
    this.trueExpression = trueExpression;
    this.ifToken = ifToken;
    this.condition = condition;
    this.elseToken = elseToken;
    this.falseExpression = falseExpression;
  }

  @Override
  public Token ifKeyword() {
    return ifToken;
  }

  @Override
  public Token elseKeyword() {
    return elseToken;
  }

  @Override
  public Expression trueExpression() {
    return trueExpression;
  }

  @Override
  public Expression falseExpression() {
    return falseExpression;
  }

  @Override
  public Expression condition() {
    return condition;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitConditionalExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(trueExpression, ifToken, condition, elseToken, falseExpression).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return Kind.CONDITIONAL_EXPR;
  }

  @Override
  public InferredType type() {
    return InferredTypes.or(trueExpression.type(), falseExpression.type());
  }

  @Override
  public List<Expression> typeDependencies() {
    return Arrays.asList(trueExpression, falseExpression);
  }
}
