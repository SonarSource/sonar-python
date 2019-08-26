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
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;

public class PyAssignmentStatementTreeImpl extends PyTree implements PyAssignmentStatementTree {
  private final List<Token> assignTokens;
  private final List<PyExpressionListTree> lhsExpressions;
  private final List<PyExpressionTree> assignedValues;

  public PyAssignmentStatementTreeImpl(AstNode astNode, List<Token> assignTokens, List<PyExpressionListTree> lhsExpressions, List<PyExpressionTree> assignedValue) {
    super(astNode);
    this.assignTokens = assignTokens;
    this.lhsExpressions = lhsExpressions;
    this.assignedValues = assignedValue;
  }

  @Override
  public List<PyExpressionTree> assignedValues() {
    return assignedValues;
  }

  @Override
  public List<Token> equalTokens() {
    return assignTokens;
  }

  @Override
  public List<PyExpressionListTree> lhsExpressions() {
    return lhsExpressions;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitAssignmentStatement(this);
  }

  @Override
  public Kind getKind() {
    return Kind.ASSIGNEMENT_STMT;
  }
}
