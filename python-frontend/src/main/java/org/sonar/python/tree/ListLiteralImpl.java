/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

public class ListLiteralImpl extends PyTree implements ListLiteral {

  private final Token leftBracket;
  private final ExpressionList elements;
  private final Token rightBracket;

  public ListLiteralImpl(Token leftBracket, ExpressionList elements, Token rightBracket) {
    this.leftBracket = leftBracket;
    this.elements = elements;
    this.rightBracket = rightBracket;
  }

  @Override
  public Kind getKind() {
    return Kind.LIST_LITERAL;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitListLiteral(this);
  }

  @Override
  public Token leftBracket() {
    return leftBracket;
  }

  @Override
  public ExpressionList elements() {
    return elements;
  }

  @Override
  public Token rightBracket() {
    return rightBracket;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(leftBracket, elements, rightBracket).collect(Collectors.toList());
  }

  @Override
  public InferredType type() {
    return InferredTypes.LIST;
  }
}
