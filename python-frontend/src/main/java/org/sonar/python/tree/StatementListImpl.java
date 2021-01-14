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
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class StatementListImpl extends PyTree implements StatementList {

  private List<Statement> statements;

  public StatementListImpl(List<Statement> statements) {
    this.statements = statements;
  }

  @Override
  public List<Statement> statements() {
    return statements;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitStatementList(this);
  }

  @Override
  public Kind getKind() {
    return Kind.STATEMENT_LIST;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(statements).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }

}
