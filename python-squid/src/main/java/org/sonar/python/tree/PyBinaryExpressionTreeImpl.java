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
import org.sonar.python.api.tree.PyBinaryExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;

public class PyBinaryExpressionTreeImpl extends PyTree implements PyBinaryExpressionTree {

  private static final Map<String, Kind> KINDS_BY_OPERATOR = kindsByOperator();

  private final Kind kind;
  private final PyExpressionTree leftOperand;
  private final Token operator;
  private final PyExpressionTree rightOperand;

  private static Map<String, Kind> kindsByOperator() {
    Map<String, Kind> map = new HashMap<>();
    map.put("+", Kind.PLUS);
    map.put("-", Kind.MINUS);
    map.put("*", Kind.MULTIPLICATION);
    map.put("/", Kind.DIVISION);
    map.put("//", Kind.FLOOR_DIVISION);
    map.put("%", Kind.MODULO);
    map.put("@", Kind.MATRIX_MULTIPLICATION);
    map.put(">>", Kind.SHIFT_EXPR);
    map.put("<<", Kind.SHIFT_EXPR);
    map.put("&", Kind.BITWISE_AND);
    map.put("|", Kind.BITWISE_OR);
    map.put("^", Kind.BITWISE_XOR);
    map.put("and", Kind.AND);
    map.put("or", Kind.OR);
    map.put("==", Kind.COMPARISON);
    map.put("<=", Kind.COMPARISON);
    map.put(">=", Kind.COMPARISON);
    map.put("<", Kind.COMPARISON);
    map.put(">", Kind.COMPARISON);
    map.put("!=", Kind.COMPARISON);
    map.put("<>", Kind.COMPARISON);
    return map;
  }

  public PyBinaryExpressionTreeImpl(AstNode node, PyExpressionTree leftOperand, Token operator, PyExpressionTree rightOperand) {
    super(node);
    this.kind = KINDS_BY_OPERATOR.get(operator.getValue());
    this.leftOperand = leftOperand;
    this.operator = operator;
    this.rightOperand = rightOperand;
  }

  @Override
  public PyExpressionTree leftOperand() {
    return leftOperand;
  }

  @Override
  public Token operator() {
    return operator;
  }

  @Override
  public PyExpressionTree rightOperand() {
    return rightOperand;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitBinaryExpression(this);
  }

  @Override
  public Kind getKind() {
    return kind;
  }
}
