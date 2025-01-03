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
package org.sonar.python.tree;

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;

public class IsExpressionImpl extends BinaryExpressionImpl implements IsExpression {

  private final Token notToken;

  public IsExpressionImpl(Expression leftOperand, Token operator, @Nullable Token not, Expression rightOperand) {
    super(leftOperand, operator, rightOperand);
    this.notToken = not;
  }

  @Override
  public Kind getKind() {
    return Kind.IS;
  }

  @CheckForNull
  @Override
  public Token notToken() {
    return notToken;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(leftOperand(), operator(), notToken, rightOperand()).filter(Objects::nonNull).toList();
  }
}
