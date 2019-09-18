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
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyAssignmentStatementTreeImpl extends PyTree implements PyAssignmentStatementTree {
  private final List<PyToken> assignTokens;
  private final List<PyExpressionListTree> lhsExpressions;
  private final PyExpressionTree assignedValue;

  public PyAssignmentStatementTreeImpl(AstNode astNode, List<PyToken> assignTokens, List<PyExpressionListTree> lhsExpressions,PyExpressionTree assignedValue) {
    super(astNode);
    this.assignTokens = assignTokens;
    this.lhsExpressions = lhsExpressions;
    this.assignedValue = assignedValue;
  }

  @Override
  public PyExpressionTree assignedValue() {
    return assignedValue;
  }

  @Override
  public List<PyToken> equalTokens() {
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
    return Kind.ASSIGNMENT_STMT;
  }

  @Override
  public List<Tree> children() {
    return Stream.of(lhsExpressions, Collections.singletonList(assignedValue)).flatMap(List::stream).collect(Collectors.toList());
  }
}
