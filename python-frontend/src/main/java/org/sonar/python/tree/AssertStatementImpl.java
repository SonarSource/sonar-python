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
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class AssertStatementImpl extends SimpleStatement implements AssertStatement {
  private final Token assertKeyword;
  private final Expression condition;
  @Nullable
  private final Expression message;
  private final Separators separators;

  public AssertStatementImpl(Token assertKeyword, Expression condition, @Nullable Expression message, Separators separators) {
    this.assertKeyword = assertKeyword;
    this.condition = condition;
    this.message = message;
    this.separators = separators;
  }

  @Override
  public Token assertKeyword() {
    return assertKeyword;
  }

  @Override
  public Expression condition() {
    return condition;
  }

  @Override
  @Nullable
  public Expression message() {
    return message;
  }

  @Override
  @Nullable
  public Token separator() {
    return separators.last();
  }

  @Override
  public Kind getKind() {
    return Kind.ASSERT_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitAssertStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Arrays.asList(assertKeyword, condition, message), separators.elements()).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
