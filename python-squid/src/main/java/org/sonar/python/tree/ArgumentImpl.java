/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.tree;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ArgumentImpl extends PyTree implements Argument {
  private final Name keywordArgument;
  private final Expression expression;
  private final Token equalToken;
  private final Token star;
  private final Token starStar;

  public ArgumentImpl(Name keywordArgument, Expression expression, Token equalToken, @Nullable Token star, @Nullable Token starStar) {
    this.keywordArgument = keywordArgument;
    this.expression = expression;
    this.equalToken = equalToken;
    this.star = star;
    this.starStar = starStar;
  }

  public ArgumentImpl(Expression expression, @Nullable Token star, @Nullable Token starStar) {
    this.keywordArgument = null;
    this.expression = expression;
    this.equalToken = null;
    this.star = star;
    this.starStar = starStar;
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

  @CheckForNull
  @Override
  public Token starToken() {
    return star;
  }

  @CheckForNull
  @Override
  public Token starStarToken() {
    return starStar;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitArgument(this);
  }

  @Override
  public Kind getKind() {
    return Kind.ARGUMENT;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(keywordArgument, equalToken, star, starStar, expression).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
