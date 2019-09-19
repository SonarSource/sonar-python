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

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyParenthesizedExpressionTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyParenthesizedExpressionTreeImpl extends PyTree implements PyParenthesizedExpressionTree {

  private final PyToken leftParenthesis;
  private final PyExpressionTree expression;
  private final PyToken rightParenthesis;

  public PyParenthesizedExpressionTreeImpl(PyToken leftParenthesis, PyExpressionTree expression, PyToken rightParenthesis) {
    super(leftParenthesis, rightParenthesis);
    this.leftParenthesis = leftParenthesis;
    this.expression = expression;
    this.rightParenthesis = rightParenthesis;
  }

  @Override
  public PyToken leftParenthesis() {
    return leftParenthesis;
  }

  @Override
  public PyExpressionTree expression() {
    return expression;
  }

  @Override
  public PyToken rightParenthesis() {
    return rightParenthesis;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitParenthesizedExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(leftParenthesis, expression, rightParenthesis).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.PARENTHESIZED;
  }
}
