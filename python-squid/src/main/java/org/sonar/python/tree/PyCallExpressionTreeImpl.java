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
import org.sonar.python.api.tree.PyToken;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.PyArgListTree;
import org.sonar.python.api.tree.PyArgumentTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyCallExpressionTreeImpl extends PyTree implements PyCallExpressionTree {
  private final PyExpressionTree callee;
  private final PyArgListTree argumentList;
  private final PyToken leftPar;
  private final PyToken rightPar;

  public PyCallExpressionTreeImpl(AstNode astNode, PyExpressionTree callee, @Nullable PyArgListTree argumentList, PyToken leftPar, PyToken rightPar) {
    super(astNode);
    this.callee = callee;
    this.argumentList = argumentList;
    this.leftPar = leftPar;
    this.rightPar = rightPar;
  }

  public PyCallExpressionTreeImpl(PyExpressionTree callee, @Nullable PyArgListTree argumentList, PyToken leftPar, PyToken rightPar) {
    super(callee.firstToken(), rightPar);
    this.callee = callee;
    this.argumentList = argumentList;
    this.leftPar = leftPar;
    this.rightPar = rightPar;
  }

  @Override
  public PyExpressionTree callee() {
    return callee;
  }

  @Override
  public PyArgListTree argumentList() {
    return argumentList;
  }

  @Override
  public List<PyArgumentTree> arguments() {
    return argumentList != null ? argumentList.arguments() : Collections.emptyList();
  }

  @Override
  public PyToken leftPar() {
    return leftPar;
  }

  @Override
  public PyToken rightPar() {
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
    return Arrays.asList(callee, argumentList);
  }
}
