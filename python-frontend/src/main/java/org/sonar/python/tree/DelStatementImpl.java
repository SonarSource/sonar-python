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
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class DelStatementImpl extends SimpleStatement implements DelStatement {
  private final Token delKeyword;
  private final List<Expression> expressionTrees;
  private final Separators separators;

  public DelStatementImpl(Token delKeyword, List<Expression> expressionTrees, Separators separators) {
    this.delKeyword = delKeyword;
    this.expressionTrees = expressionTrees;
    this.separators = separators;
  }

  @Override
  public Token delKeyword() {
    return delKeyword;
  }

  @Override
  public List<Expression> expressions() {
    return expressionTrees;
  }

  @Nullable
  @Override
  public Token separator() {
    return separators.last();
  }

  @Override
  public Kind getKind() {
    return Kind.DEL_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitDelStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(delKeyword), expressionTrees, separators.elements())
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
