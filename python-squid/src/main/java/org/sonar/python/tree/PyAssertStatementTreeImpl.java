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
import java.util.List;
import org.sonar.python.api.tree.PyAssertStatementTree;
import org.sonar.python.api.tree.PyExpressionTree;

public class PyAssertStatementTreeImpl extends PyTree implements PyAssertStatementTree {
  private final Token assertKeyword;
  private final List<PyExpressionTree> expressions;

  public PyAssertStatementTreeImpl(AstNode astNode, Token assertKeyword, List<PyExpressionTree> expressions) {
    super(astNode);
    this.assertKeyword = assertKeyword;
    this.expressions = expressions;
  }

  @Override
  public Token assertKeyword() {
    return assertKeyword;
  }

  @Override
  public List<PyExpressionTree> expressions() {
    return expressions;
  }

  @Override
  public Kind getKind() {
    return Kind.ASSERT_STMT;
  }
}
