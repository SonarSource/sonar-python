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

import com.sonar.sslr.api.AstNode;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyArgumentTreeImpl extends PyTree implements PyArgumentTree {
  private final PyNameTree keywordArgument;
  private final PyExpressionTree expression;
  private final PyToken equalToken;
  private final PyToken star;
  private final PyToken starStar;

  public PyArgumentTreeImpl(AstNode node, PyNameTree keywordArgument, PyExpressionTree expression, PyToken equalToken, @Nullable PyToken star, @Nullable PyToken starStar) {
    super(node);
    this.keywordArgument = keywordArgument;
    this.expression = expression;
    this.equalToken = equalToken;
    this.star = star;
    this.starStar = starStar;
  }

  public PyArgumentTreeImpl(AstNode astNode, PyExpressionTree expression, @Nullable PyToken star, @Nullable PyToken starStar) {
    super(astNode);
    this.keywordArgument = null;
    this.expression = expression;
    this.equalToken = null;
    this.star = star;
    this.starStar = starStar;
  }

  @CheckForNull
  @Override
  public PyNameTree keywordArgument() {
    return keywordArgument;
  }

  @CheckForNull
  @Override
  public PyToken equalToken() {
    return equalToken;
  }

  @Override
  public PyExpressionTree expression() {
    return expression;
  }

  @CheckForNull
  @Override
  public PyToken starToken() {
    return star;
  }

  @CheckForNull
  @Override
  public PyToken starStarToken() {
    return starStar;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitArgument(this);
  }

  @Override
  public Kind getKind() {
    return Kind.ARGUMENT;
  }

  @Override
  public List<Tree> children() {
    return Stream.of(keywordArgument, expression, equalToken, star, starStar).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
