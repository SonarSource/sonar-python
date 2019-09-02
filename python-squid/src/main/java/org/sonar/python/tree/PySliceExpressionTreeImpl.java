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
import java.util.Arrays;
import java.util.List;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PySliceExpressionTree;
import org.sonar.python.api.tree.PySliceListTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PySliceExpressionTreeImpl extends PyTree implements PySliceExpressionTree {

  private final PyExpressionTree object;
  private final Token leftBracket;
  private final PySliceListTree sliceList;
  private final Token rightBracket;

  public PySliceExpressionTreeImpl(PyExpressionTree object, Token leftBracket, PySliceListTree sliceList, Token rightBracket) {
    super(object.firstToken(), rightBracket);
    this.object = object;
    this.leftBracket = leftBracket;
    this.sliceList = sliceList;
    this.rightBracket = rightBracket;
  }

  @Override
  public PyExpressionTree object() {
    return object;
  }

  @Override
  public Token leftBracket() {
    return leftBracket;
  }

  @Override
  public PySliceListTree sliceList() {
    return sliceList;
  }

  @Override
  public Token rightBracket() {
    return rightBracket;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitSliceExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(object, sliceList);
  }

  @Override
  public Kind getKind() {
    return Kind.SLICE_EXPR;
  }
}
