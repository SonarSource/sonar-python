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

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.YieldExpression;

public class YieldExpressionImpl extends PyTree implements YieldExpression {
  private final Token yieldKeyword;
  private final Token fromKeyword;
  private final List<Expression> expressionTrees;

  public YieldExpressionImpl(Token yieldKeyword, @Nullable Token fromKeyword, List<Expression> expressionTrees) {
    this.yieldKeyword = yieldKeyword;
    this.fromKeyword = fromKeyword;
    this.expressionTrees = expressionTrees;
  }

  @Override
  public Token yieldKeyword() {
    return yieldKeyword;
  }

  @CheckForNull
  @Override
  public Token fromKeyword() {
    return fromKeyword;
  }

  @Override
  public List<Expression> expressions() {
    return expressionTrees;
  }

  @Override
  public Kind getKind() {
    return Kind.YIELD_EXPR;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitYieldExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Arrays.asList(yieldKeyword, fromKeyword), expressionTrees)
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
