/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class AssignmentStatementImpl extends SimpleStatement implements AssignmentStatement {
  private final List<Token> assignTokens;
  private final List<ExpressionList> lhsExpressions;
  private final Expression assignedValue;
  private final Separators separators;

  public AssignmentStatementImpl(List<Token> assignTokens, List<ExpressionList> lhsExpressions, Expression assignedValue, Separators separators) {
    this.assignTokens = assignTokens;
    this.lhsExpressions = lhsExpressions;
    this.assignedValue = assignedValue;
    this.separators = separators;
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
    return separators.last();
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
  public List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    int i = 0;
    for (Tree lhs : lhsExpressions) {
      children.add(lhs);
      if (i < assignTokens.size()) {
        children.add(assignTokens.get(i));
      }
      i++;
    }
    children.add(assignedValue);
    children.addAll(separators.elements());
    return children;
  }
}
