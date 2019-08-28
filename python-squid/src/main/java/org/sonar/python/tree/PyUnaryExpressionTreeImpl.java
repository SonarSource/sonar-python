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
import java.util.HashMap;
import java.util.Map;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyUnaryExpressionTree;

public class PyUnaryExpressionTreeImpl extends PyTree implements PyUnaryExpressionTree {

  private static final Map<String, Kind> KINDS_BY_OPERATOR = kindsByOperator();

  private final Kind kind;
  private final Token operator;
  private final PyExpressionTree expression;

  private static Map<String, Kind> kindsByOperator() {
    Map<String, Kind> map = new HashMap<>();
    map.put("+", Kind.UNARY_PLUS);
    map.put("-", Kind.UNARY_MINUS);
    map.put("~", Kind.BITWISE_COMPLEMENT);
    return map;
  }

  public PyUnaryExpressionTreeImpl(AstNode node, Token operator, PyExpressionTree expression) {
    super(node);
    this.kind = KINDS_BY_OPERATOR.get(operator.getValue());
    this.operator = operator;
    this.expression = expression;
  }

  @Override
  public Token operator() {
    return operator;
  }

  @Override
  public PyExpressionTree expression() {
    return expression;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitUnaryExpression(this);
  }

  @Override
  public Kind getKind() {
    return kind;
  }
}
