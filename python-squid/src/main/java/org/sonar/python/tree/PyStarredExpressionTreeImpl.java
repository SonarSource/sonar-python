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
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyStarredExpressionTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyStarredExpressionTreeImpl extends PyTree implements PyStarredExpressionTree {

  private final PyToken starToken;
  private final PyExpressionTree expression;

  public PyStarredExpressionTreeImpl(AstNode node, PyToken starToken, PyExpressionTree expression) {
    super(node);
    this.starToken = starToken;
    this.expression = expression;
  }

  @Override
  public PyToken starToken() {
    return starToken;
  }

  @Override
  public PyExpressionTree expression() {
    return expression;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitStarredExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(starToken, expression).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.STARRED_EXPR;
  }
}
