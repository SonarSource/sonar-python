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
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

/**
 * Even if decorator is not a statement, it extends {@code SimpleStatement} in order to properly resolve
 * its last token as it's a newline token
 */
public class DecoratorImpl extends SimpleStatement implements Decorator {
  private final Token atToken;
  private final Token newLineToken;
  private final Expression expression;

  public DecoratorImpl(Token atToken, Expression expression, @Nullable Token newLineToken) {
    this.atToken = atToken;
    this.expression = expression;
    this.newLineToken = newLineToken != null ? newLineToken : null;
  }

  @Override
  public Token atToken() {
    return atToken;
  }

  @CheckForNull
  @Override
  public ArgList arguments() {
    if (expression.is(Kind.CALL_EXPR)) {
      return ((CallExpression) expression).argumentList();
    }
    return null;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitDecorator(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(atToken, expression, newLineToken).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return Kind.DECORATOR;
  }
}
