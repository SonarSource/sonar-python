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
import org.sonar.plugins.python.api.tree.ReprExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ReprExpressionImpl extends PyTree implements ReprExpression {
  private final Token openingBacktick;
  private final ExpressionList expressionListTree;
  private final Token closingBacktick;

  public ReprExpressionImpl(Token openingBacktick, ExpressionList expressionListTree, Token closingBacktick) {
    this.openingBacktick = openingBacktick;
    this.expressionListTree = expressionListTree;
    this.closingBacktick = closingBacktick;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitRepr(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(openingBacktick, expressionListTree, closingBacktick).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.REPR;
  }

  @Override
  public Token openingBacktick() {
    return openingBacktick;
  }

  @Override
  public ExpressionList expressionList() {
    return expressionListTree;
  }

  @Override
  public Token closingBacktick() {
    return closingBacktick;
  }
}
