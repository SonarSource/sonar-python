/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ComprehensionIfImpl extends PyTree implements ComprehensionIf {

  private final Token ifToken;
  private final Expression condition;
  private final ComprehensionClause nestedClause;

  public ComprehensionIfImpl(Token ifToken, Expression condition, @Nullable ComprehensionClause nestedClause) {
    this.ifToken = ifToken;
    this.condition = condition;
    this.nestedClause = nestedClause;
  }

  @Override
  public Token ifToken() {
    return ifToken;
  }

  @Override
  public Expression condition() {
    return condition;
  }

  @CheckForNull
  @Override
  public ComprehensionClause nestedClause() {
    return nestedClause;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitComprehensionIf(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(ifToken, condition, nestedClause).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public Kind getKind() {
    return Kind.COMP_IF;
  }
}
