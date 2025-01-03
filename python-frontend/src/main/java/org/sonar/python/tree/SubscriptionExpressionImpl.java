/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.python.types.v2.PythonType;

public class SubscriptionExpressionImpl extends PyTree implements SubscriptionExpression {

  private final Expression object;
  private final Token lBracket;
  private final ExpressionList subscripts;
  private final Token rBracket;
  private PythonType pythonType = PythonType.UNKNOWN;

  public SubscriptionExpressionImpl(Expression object, Token lBracket, ExpressionList subscripts, Token rBracket) {
    this.object = object;
    this.lBracket = lBracket;
    this.subscripts = subscripts;
    this.rBracket = rBracket;
  }

  @Override
  public Expression object() {
    return object;
  }

  @Override
  public Token leftBracket() {
    return lBracket;
  }

  @Override
  public ExpressionList subscripts() {
    return subscripts;
  }

  @Override
  public Token rightBracket() {
    return rBracket;
  }

  @Override
  public PythonType typeV2() {
    return pythonType;
  }

  public SubscriptionExpression typeV2(PythonType pythonType) {
    this.pythonType = pythonType;
    return this;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitSubscriptionExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(object, lBracket, subscripts, rBracket).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return Kind.SUBSCRIPTION;
  }
}
