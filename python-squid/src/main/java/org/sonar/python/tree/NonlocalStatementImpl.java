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

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.NonlocalStatement;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TreeVisitor;

public class NonlocalStatementImpl extends PyTree implements NonlocalStatement {
  private final Token nonlocalKeyword;
  private final List<Name> variables;
  private final Token separator;

  public NonlocalStatementImpl(Token nonlocalKeyword, List<Name> variables, @Nullable Token separator) {
    super(nonlocalKeyword, variables.get(variables.size() - 1).lastToken());
    this.nonlocalKeyword = nonlocalKeyword;
    this.variables = variables;
    this.separator = separator;
  }

  @Override
  public Token nonlocalKeyword() {
    return nonlocalKeyword;
  }

  @Override
  public List<Name> variables() {
    return variables;
  }

  @CheckForNull
  @Override
  public Token separator() {
    return separator;
  }

  @Override
  public Kind getKind() {
    return Kind.NONLOCAL_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitNonlocalStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(nonlocalKeyword), variables, Collections.singletonList(separator))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
