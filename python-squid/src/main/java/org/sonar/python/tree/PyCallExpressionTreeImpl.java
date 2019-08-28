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
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyCallExpressionTreeImpl extends PyExpressionTreeImpl implements PyCallExpressionTree {
  private final PyExpressionTree callee;
  private final List<PyArgumentTree> arguments;
  private final Token leftPar;
  private final Token rightPar;

  public PyCallExpressionTreeImpl(AstNode astNode, PyExpressionTree callee, List<PyArgumentTree> arguments, AstNode leftPar, AstNode rightPar) {
    super(astNode);
    this.callee = callee;
    this.arguments = arguments;
    this.leftPar = leftPar.getToken();
    this.rightPar = rightPar.getToken();
  }

  @Override
  public PyExpressionTree callee() {
    return callee;
  }

  @Override
  public List<PyArgumentTree> arguments() {
    return arguments;
  }

  @Override
  public Token leftPar() {
    return leftPar;
  }

  @Override
  public Token rightPar() {
    return rightPar;
  }

  @Override
  public Kind getKind() {
    return Kind.CALL_EXPR;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitCallExpression(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(Collections.singletonList(callee), arguments).flatMap(List::stream).collect(Collectors.toList());
  }
}
