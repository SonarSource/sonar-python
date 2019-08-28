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
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyRaiseStatementTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyRaiseStatementTreeImpl extends PyTree implements PyRaiseStatementTree {
  private final Token raiseKeyword;
  private final List<PyExpressionTree> expressions;
  private final Token fromKeyword;
  private final PyExpressionTree fromExpression;

  public PyRaiseStatementTreeImpl(AstNode astNode, Token raiseKeyword, List<PyExpressionTree> expressions, Token fromKeyword, PyExpressionTree fromExpression) {
    super(astNode);
    this.raiseKeyword = raiseKeyword;
    this.expressions = expressions;
    this.fromKeyword = fromKeyword;
    this.fromExpression = fromExpression;
  }

  @Override
  public Token raiseKeyword() {
    return raiseKeyword;
  }

  @CheckForNull
  @Override
  public Token fromKeyword() {
    return fromKeyword;
  }

  @CheckForNull
  @Override
  public PyExpressionTree fromExpression() {
    return fromExpression;
  }

  @Override
  public List<PyExpressionTree> expressions() {
    return expressions;
  }

  @Override
  public Kind getKind() {
    return Kind.RAISE_STMT;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitRaiseStatement(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(expressions, Collections.singletonList(fromExpression))
      .flatMap(List::stream).collect(Collectors.toList());
  }
}
