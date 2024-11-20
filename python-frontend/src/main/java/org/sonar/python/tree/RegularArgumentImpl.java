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
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class RegularArgumentImpl extends PyTree implements RegularArgument {
  private final Name keywordArgument;
  private final Expression expression;
  private final Token equalToken;

  public RegularArgumentImpl(Name keywordArgument, Token equalToken, Expression expression) {
    this.keywordArgument = keywordArgument;
    this.equalToken = equalToken;
    this.expression = expression;
  }

  public RegularArgumentImpl(Expression expression) {
    this.keywordArgument = null;
    this.equalToken = null;
    this.expression = expression;
  }

  @CheckForNull
  @Override
  public Name keywordArgument() {
    return keywordArgument;
  }

  @CheckForNull
  @Override
  public Token equalToken() {
    return equalToken;
  }

  @Override
  public Expression expression() {
    return expression;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitRegularArgument(this);
  }

  @Override
  public Kind getKind() {
    return Kind.REGULAR_ARGUMENT;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(keywordArgument, equalToken, expression).filter(Objects::nonNull).toList();
  }
}
