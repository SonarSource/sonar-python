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

import java.util.List;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ReprExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ReprExpressionImpl extends PyTree implements ReprExpression {
  private final Token openingBacktick;
  private final ExpressionList expressionListTree;
  private final Token closingBacktick;

  public ReprExpressionImpl(Token openingBacktick, ExpressionList expressionListTree, Token closingBacktick) {
    this.openingBacktick = openingBacktick;
    this.expressionListTree = expressionListTree;
    this.closingBacktick = closingBacktick;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitRepr(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(openingBacktick, expressionListTree, closingBacktick).toList();
  }

  @Override
  public Kind getKind() {
    return Kind.REPR;
  }

  @Override
  public Token openingBacktick() {
    return openingBacktick;
  }

  @Override
  public ExpressionList expressionList() {
    return expressionListTree;
  }

  @Override
  public Token closingBacktick() {
    return closingBacktick;
  }
}
