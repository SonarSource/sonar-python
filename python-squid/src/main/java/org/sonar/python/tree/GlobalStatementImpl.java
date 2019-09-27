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

import com.sonar.sslr.api.AstNode;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.python.api.tree.GlobalStatement;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.Token;
import org.sonar.python.api.tree.TreeVisitor;
import org.sonar.python.api.tree.Tree;

public class GlobalStatementImpl extends PyTree implements GlobalStatement {
  private final Token globalKeyword;
  private final List<Name> variables;

  public GlobalStatementImpl(AstNode astNode, Token globalKeyword, List<Name> variables) {
    super(astNode);
    this.globalKeyword = globalKeyword;
    this.variables = variables;
  }

  @Override
  public Token globalKeyword() {
    return globalKeyword;
  }

  @Override
  public List<Name> variables() {
    return variables;
  }

  @Override
  public Kind getKind() {
    return Kind.GLOBAL_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitGlobalStatement(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(Collections.singletonList(globalKeyword), variables).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
