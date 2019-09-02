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
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PySubscriptionExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PySubscriptionExpressionTreeImpl extends PyTree implements PySubscriptionExpressionTree {

  private final PyExpressionTree object;
  private final Token lBracket;
  private final PyExpressionListTree subscripts;
  private final Token rBracket;

  public PySubscriptionExpressionTreeImpl(PyExpressionTree object, Token lBracket, PyExpressionListTree subscripts, Token rBracket) {
    super(object.firstToken(), rBracket);
    this.object = object;
    this.lBracket = lBracket;
    this.subscripts = subscripts;
    this.rBracket = rBracket;
  }

  @Override
  public PyExpressionTree object() {
    return object;
  }

  @Override
  public Token leftBracket() {
    return lBracket;
  }

  @Override
  public PyExpressionListTree subscripts() {
    return subscripts;
  }

  @Override
  public Token rightBracket() {
    return rBracket;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitSubscriptionExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(object, subscripts);
  }

  @Override
  public Kind getKind() {
    return Kind.SUBSCRIPTION;
  }
}
