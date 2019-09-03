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
import java.util.Arrays;
import java.util.List;
import org.sonar.python.api.tree.PyConditionalExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyConditionalExpressionTreeImpl extends PyTree implements PyConditionalExpressionTree {
  private final PyExpressionTree trueExpression;
  private final Token ifToken;
  private final PyExpressionTree condition;
  private final Token elseToken;
  private final PyExpressionTree falseExpression;

  public PyConditionalExpressionTreeImpl(AstNode node, PyExpressionTree trueExpression, Token ifToken, PyExpressionTree condition, Token elseToken, PyExpressionTree falseExpression) {
    super(node);
    this.trueExpression = trueExpression;
    this.ifToken = ifToken;
    this.condition = condition;
    this.elseToken = elseToken;
    this.falseExpression = falseExpression;
  }

  @Override
  public Token ifKeyword() {
    return ifToken;
  }

  @Override
  public Token elseKeyword() {
    return elseToken;
  }

  @Override
  public PyExpressionTree trueExpression() {
    return trueExpression;
  }

  @Override
  public PyExpressionTree falseExpression() {
    return falseExpression;
  }

  @Override
  public PyExpressionTree condition() {
    return condition;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitConditionalExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(condition, trueExpression, falseExpression);
  }

  @Override
  public Kind getKind() {
    return Kind.CONDITIONAL_EXPR;
  }
}
