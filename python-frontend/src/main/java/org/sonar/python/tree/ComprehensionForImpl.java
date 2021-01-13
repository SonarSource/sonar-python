/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import org.sonar.plugins.python.api.tree.ComprehensionClause;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ComprehensionForImpl extends PyTree implements ComprehensionFor {

  private final Token asyncToken;
  private final Token forToken;
  private final Expression loopExpression;
  private final Token inToken;
  private final Expression iterable;
  private final ComprehensionClause nested;

  public ComprehensionForImpl(@Nullable Token asyncToken, Token forToken, Expression loopExpression, Token inToken,
                              Expression iterable, @Nullable ComprehensionClause nested) {
    this.asyncToken = asyncToken;
    this.forToken = forToken;
    this.loopExpression = loopExpression;
    this.inToken = inToken;
    this.iterable = iterable;
    this.nested = nested;
  }

  @Override
  public Token asyncToken() {
    return asyncToken;
  }

  @Override
  public Token forToken() {
    return forToken;
  }

  @Override
  public Expression loopExpression() {
    return loopExpression;
  }

  @Override
  public Token inToken() {
    return inToken;
  }

  @Override
  public Expression iterable() {
    return iterable;
  }

  @CheckForNull
  @Override
  public ComprehensionClause nestedClause() {
    return nested;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitComprehensionFor(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(asyncToken, forToken, loopExpression, inToken, iterable, nested).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.COMP_FOR;
  }
}
