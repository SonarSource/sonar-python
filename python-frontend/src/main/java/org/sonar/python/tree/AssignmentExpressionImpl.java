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

import java.util.Arrays;
import java.util.List;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class AssignmentExpressionImpl extends PyTree implements AssignmentExpression {

  private final Name name;
  private final Token walrusOperator;
  private final Expression expression;

  public AssignmentExpressionImpl(Name name, Token walrusOperator, Expression expression) {
    this.name = name;
    this.walrusOperator = walrusOperator;
    this.expression = expression;
  }

  @Override
  public Name lhsName() {
    return name;
  }

  @Override
  public Token operator() {
    return walrusOperator;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitAssignmentExpression(this);
  }

  @Override
  public Kind getKind() {
    return Kind.ASSIGNMENT_EXPRESSION;
  }

  @Override
  List<Tree> computeChildren() {
    return Arrays.asList(name, walrusOperator, expression);
  }
}
