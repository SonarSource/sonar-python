/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import com.sonar.sslr.api.Token;
import java.util.Collections;
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyTypeAnnotationTree;
import org.sonar.python.api.tree.Tree;

public class PyTypeAnnotationTreeImpl extends PyTree implements PyTypeAnnotationTree {

  private final Token dash;
  private final Token gt;
  private final Token colonToken;
  private final PyExpressionTree expression;
  private final Kind kind;

  public PyTypeAnnotationTreeImpl(Token colonToken, PyExpressionTree expression) {
    super(colonToken, expression.lastToken());
    this.colonToken = colonToken;
    this.dash = null;
    this.gt = null;
    this.expression = expression;
    this.kind = Kind.TYPE_ANNOTATION;
  }

  public PyTypeAnnotationTreeImpl(Token dash, Token gt, PyExpressionTree expression) {
    super(dash, expression.lastToken());
    this.colonToken = null;
    this.dash = dash;
    this.gt = gt;
    this.expression = expression;
    this.kind = Kind.RETURN_TYPE_ANNOTATION;
  }

  @CheckForNull
  @Override
  public Token colonToken() {
    return colonToken;
  }

  @CheckForNull
  @Override
  public Token dash() {
    return dash;
  }

  @CheckForNull
  @Override
  public Token gt() {
    return gt;
  }

  @Override
  public PyExpressionTree expression() {
    return expression;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitTypeAnnotation(this);
  }

  @Override
  public List<Tree> children() {
    return Collections.singletonList(expression);
  }

  @Override
  public Kind getKind() {
    return kind;
  }
}
