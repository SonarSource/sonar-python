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
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyLambdaExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.PyParameterListTree;
import org.sonar.python.api.tree.Tree;

public class PyLambdaExpressionTreeImpl extends PyExpressionTreeImpl implements PyLambdaExpressionTree {
  private final Token lambdaKeyword;
  private final Token colonToken;
  private final PyExpressionTree body;
  private final PyParameterListTree parameterList;

  public PyLambdaExpressionTreeImpl(AstNode astNode, Token lambdaKeyword, Token colonToken, PyExpressionTree body, PyParameterListTree parameterList) {
    super(astNode);
    this.lambdaKeyword = lambdaKeyword;
    this.colonToken = colonToken;
    this.body = body;
    this.parameterList = parameterList;
  }

  @Override
  public Token lambdaKeyword() {
    return lambdaKeyword;
  }

  @Override
  public Token colonToken() {
    return colonToken;
  }

  @Override
  public PyExpressionTree expression() {
    return body;
  }

  @Override
  public PyParameterListTree arguments() {
    return parameterList;
  }

  @Override
  public Kind getKind() {
    return Kind.LAMBDA;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitLambda(this);
  }

  @Override
  public List<Tree> children() {
    return Arrays.asList(parameterList, body);
  }
}
