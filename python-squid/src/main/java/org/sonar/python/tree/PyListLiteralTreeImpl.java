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
import com.sonar.sslr.api.Token;
import java.util.Collections;
import java.util.List;
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyListLiteralTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyListLiteralTreeImpl extends PyTree implements PyListLiteralTree {

  private final Token leftBracket;
  private final PyExpressionListTree elements;
  private final Token rightBracket;

  public PyListLiteralTreeImpl(AstNode astNode, Token leftBracket, PyExpressionListTree elements, Token rightBracket) {
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
  public Token leftBracket() {
    return leftBracket;
  }

  @Override
  public PyExpressionListTree elements() {
    return elements;
  }

  @Override
  public Token rightBracket() {
    return rightBracket;
  }

  @Override
  public List<Tree> children() {
    return Collections.singletonList(elements);
  }
}
