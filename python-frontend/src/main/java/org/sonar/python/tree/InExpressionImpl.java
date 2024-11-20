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
package org.sonar.python.tree;

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.InExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;

public class InExpressionImpl extends BinaryExpressionImpl implements InExpression {

  private final Token notToken;

  public InExpressionImpl(Expression leftOperand, @Nullable Token not, Token operator, Expression rightOperand) {
    super(leftOperand, operator, rightOperand);
    this.notToken = not;
  }

  @Override
  public Kind getKind() {
    return Kind.IN;
  }

  @CheckForNull
  @Override
  public Token notToken() {
    return notToken;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(leftOperand(), notToken, operator(), rightOperand()).filter(Objects::nonNull).toList();
  }
}
