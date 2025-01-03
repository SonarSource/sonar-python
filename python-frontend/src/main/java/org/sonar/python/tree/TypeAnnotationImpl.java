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
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.TypeAnnotation;

public class TypeAnnotationImpl extends PyTree implements TypeAnnotation {

  private final Token dash;
  private final Token gt;
  private final Token colonToken;
  private final Token starToken;
  private final Expression expression;
  private final Kind kind;

  public TypeAnnotationImpl(Token colonToken, @Nullable Token starToken, Expression expression, Kind kind) {
    this.colonToken = colonToken;
    this.dash = null;
    this.gt = null;
    this.starToken = starToken;
    this.expression = expression;
    this.kind = kind;
  }

  public TypeAnnotationImpl(Token dash, Token gt, Expression expression) {
    this.colonToken = null;
    this.dash = dash;
    this.gt = gt;
    this.starToken = null;
    this.expression = expression;
    this.kind = Kind.RETURN_TYPE_ANNOTATION;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitTypeAnnotation(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(dash, gt, colonToken, starToken, expression).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return kind;
  }
}
