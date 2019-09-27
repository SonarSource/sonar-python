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
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.AssignmentStatement;
import org.sonar.python.api.tree.Expression;
import org.sonar.python.api.tree.ExpressionList;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TreeVisitor;

public class AssignmentStatementImpl extends PyTree implements AssignmentStatement {
  private final List<Token> assignTokens;
  private final List<ExpressionList> lhsExpressions;
  private final Expression assignedValue;
  private final Token separator;

  public AssignmentStatementImpl(AstNode astNode, List<Token> assignTokens, List<ExpressionList> lhsExpressions, Expression assignedValue, @Nullable Token separator) {
    super(astNode);
    this.assignTokens = assignTokens;
    this.lhsExpressions = lhsExpressions;
    this.assignedValue = assignedValue;
    this.separator = separator;
  }

  @Override
  public Expression assignedValue() {
    return assignedValue;
  }

  @Override
  public List<Token> equalTokens() {
    return assignTokens;
  }

  @Override
  public List<ExpressionList> lhsExpressions() {
    return lhsExpressions;
  }

  @CheckForNull
  @Override
  public Token separator() {
    return separator;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitAssignmentStatement(this);
  }

  @Override
  public Kind getKind() {
    return Kind.ASSIGNMENT_STMT;
  }

  @Override
  public List<Tree> children() {
    return Stream.of(assignTokens, lhsExpressions, Collections.singletonList(assignedValue), Collections.singletonList(separator))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
