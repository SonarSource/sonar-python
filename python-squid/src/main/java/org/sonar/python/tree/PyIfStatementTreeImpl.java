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
import com.sonar.sslr.api.Token;
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyElseStatementTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.PyTree;

public class PyIfStatementTreeImpl extends PyTree implements PyIfStatementTree {

  private final Token keyword;
  private final PyExpressionTree condition;
  private final List<PyStatementTree> statements;

  public PyIfStatementTreeImpl(AstNode node, Token keyword, PyExpressionTree condition, List<PyStatementTree> statements) {
    super(node);
    this.keyword = keyword;
    this.condition = condition;
    this.statements = statements;
  }

  @Override
  public Token keyword() {
    return keyword;
  }

  @Override
  public PyExpressionTree condition() {
    return condition;
  }

  @Override
  public List<PyStatementTree> body() {
    return null;
  }

  @Override
  public List<PyIfStatementTree> elifBranches() {
    return null;
  }

  @Override
  public boolean isElif() {
    return false;
  }

  @CheckForNull
  @Override
  public PyElseStatementTree elseBranch() {
    return null;
  }

  @Override
  public Kind getKind() {
    return null;
  }
}
