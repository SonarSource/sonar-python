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

import java.util.ArrayList;
import java.util.List;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.ExpressionList;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TreeVisitor;

public class ExpressionListImpl extends PyTree implements ExpressionList {
  private final List<Expression> expressions;
  private final List<Token> commas;

  public ExpressionListImpl(List<Expression> expressions, List<Token> commas) {
    super(expressions.isEmpty() ? null : expressions.get(0).firstToken(),
      expressions.isEmpty() ? null : expressions.get(expressions.size() - 1).lastToken());
    this.expressions = expressions;
    this.commas = commas;
  }

  @Override
  public List<Expression> expressions() {
    return expressions;
  }

  @Override
  public List<Token> commas() {
    return commas;
  }

  @Override
  public Kind getKind() {
    return Kind.EXPRESSION_LIST;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitExpressionList(this);
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    int i = 0;
    for (Expression expression : expressions) {
      children.add(expression);
      if (i < commas.size()) {
        children.add(commas.get(i));
      }
      i++;
    }
    return children;
  }
}
