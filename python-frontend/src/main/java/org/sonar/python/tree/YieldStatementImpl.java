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

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.YieldExpression;
import org.sonar.plugins.python.api.tree.YieldStatement;

public class YieldStatementImpl extends SimpleStatement implements YieldStatement {
  private final YieldExpression yieldExpression;
  private final Separators separators;

  public YieldStatementImpl(YieldExpression yieldExpression, Separators separators) {
    this.yieldExpression = yieldExpression;
    this.separators = separators;
  }

  @Override
  public YieldExpression yieldExpression() {
    return yieldExpression;
  }

  @Override
  public Token separator() {
    return separators.last();
  }

  @Override
  public Kind getKind() {
    return Kind.YIELD_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitYieldStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(yieldExpression), separators.elements()).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
