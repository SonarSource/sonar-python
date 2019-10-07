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
import javax.annotation.Nullable;
import org.sonar.python.api.tree.ElseStatement;
import org.sonar.python.api.tree.StatementList;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.TreeVisitor;

public class ElseStatementImpl extends PyTree implements ElseStatement {
  private final Token elseKeyword;
  private final Token colon;
  private final Token newLine;
  private final Token indent;
  private final StatementList body;
  private final Token dedent;

  public ElseStatementImpl(Token elseKeyword, Token colon, @Nullable Token newLine, @Nullable Token indent, StatementList body, @Nullable Token dedent) {
    super(elseKeyword, body.lastToken());
    this.elseKeyword = elseKeyword;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.body = body;
    this.dedent = dedent;
  }

  @Override
  public Kind getKind() {
    return Tree.Kind.ELSE_STMT;
  }

  @Override
  public Token elseKeyword() {
    return elseKeyword;
  }

  @Override
  public StatementList body() {
    return body;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitElseStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(elseKeyword, colon, newLine, indent, body, dedent).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
