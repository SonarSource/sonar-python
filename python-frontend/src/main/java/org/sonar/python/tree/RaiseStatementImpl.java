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
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class RaiseStatementImpl extends SimpleStatement implements RaiseStatement {
  private final Token raiseKeyword;
  private final List<Expression> expressions;
  private final Token fromKeyword;
  private final Expression fromExpression;
  private final Separators separators;

  public RaiseStatementImpl(Token raiseKeyword, List<Expression> expressions,
                            @Nullable Token fromKeyword, @Nullable Expression fromExpression, Separators separators) {
    this.raiseKeyword = raiseKeyword;
    this.expressions = expressions;
    this.fromKeyword = fromKeyword;
    this.fromExpression = fromExpression;
    this.separators = separators;
  }

  @Override
  public Token raiseKeyword() {
    return raiseKeyword;
  }

  @CheckForNull
  @Override
  public Token fromKeyword() {
    return fromKeyword;
  }

  @CheckForNull
  @Override
  public Expression fromExpression() {
    return fromExpression;
  }

  @Override
  public List<Expression> expressions() {
    return expressions;
  }

  @CheckForNull
  @Override
  public Token separator() {
    return separators.last();
  }

  @Override
  public Kind getKind() {
    return Kind.RAISE_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitRaiseStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(raiseKeyword), expressions, Arrays.asList(fromKeyword, fromExpression), separators.elements())
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
