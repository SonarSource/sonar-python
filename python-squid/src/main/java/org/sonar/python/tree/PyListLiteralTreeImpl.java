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
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyListLiteralTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyListLiteralTreeImpl extends PyTree implements PyListLiteralTree {

  private final PyToken leftBracket;
  private final PyExpressionListTree elements;
  private final PyToken rightBracket;

  public PyListLiteralTreeImpl(AstNode astNode, PyToken leftBracket, PyExpressionListTree elements, PyToken rightBracket) {
    super(astNode);
    this.leftBracket = leftBracket;
    this.elements = elements;
    this.rightBracket = rightBracket;
  }

  @Override
  public Kind getKind() {
    return Kind.LIST_LITERAL;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitListLiteral(this);
  }

  @Override
  public PyToken leftBracket() {
    return leftBracket;
  }

  @Override
  public PyExpressionListTree elements() {
    return elements;
  }

  @Override
  public PyToken rightBracket() {
    return rightBracket;
  }

  @Override
  public List<Tree> children() {
    return Stream.of(leftBracket, elements, rightBracket).collect(Collectors.toList());
  }
}
