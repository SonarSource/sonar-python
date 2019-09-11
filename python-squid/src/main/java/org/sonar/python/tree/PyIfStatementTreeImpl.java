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

import com.sonar.sslr.api.Token;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.python.api.tree.PyElseStatementTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyIfStatementTree;
import org.sonar.python.api.tree.PyStatementListTree;
import org.sonar.python.api.tree.PyTreeVisitor;
import org.sonar.python.api.tree.Tree;

public class PyIfStatementTreeImpl extends PyTree implements PyIfStatementTree {

  private final Token keyword;
  private final PyExpressionTree condition;
  private final PyStatementListTree statements;
  private final List<PyIfStatementTree> elifBranches;
  private final boolean isElif;
  @CheckForNull
  private final PyElseStatementTree elseStatement;

  /**
   *
   * If statement constructor
   */
  public PyIfStatementTreeImpl(Token ifKeyword, PyExpressionTree condition, PyStatementListTree statements, List<PyIfStatementTree> elifBranches, @CheckForNull PyElseStatementTree elseStatement) {
    super(ifKeyword, statements.lastToken());
    this.keyword = ifKeyword;
    this.condition = condition;
    this.statements = statements;
    this.elifBranches = elifBranches;
    this.isElif = false;
    this.elseStatement = elseStatement;
  }

  /**
   * Elif statement constructor
   */
  public PyIfStatementTreeImpl(Token elifKeyword, PyExpressionTree condition, PyStatementListTree statements) {
    super(elifKeyword, statements.lastToken());
    this.keyword = elifKeyword;
    this.condition = condition;
    this.statements = statements;
    this.elifBranches = Collections.emptyList();
    this.isElif = true;
    this.elseStatement = null;
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
  public PyStatementListTree body() {
    return statements;
  }

  @Override
  public List<PyIfStatementTree> elifBranches() {
    return elifBranches;
  }

  @Override
  public boolean isElif() {
    return isElif;
  }

  @CheckForNull
  @Override
  public PyElseStatementTree elseBranch() {
    return elseStatement;
  }

  @Override
  public Kind getKind() {
    return Tree.Kind.IF_STMT;
  }

  @Override
  public void accept(PyTreeVisitor visitor) {
    visitor.visitIfStatement(this);
  }

  @Override
  public List<Tree> children() {
    return Stream.of(elifBranches, Arrays.asList(condition, statements, elseStatement))
      .flatMap(List::stream).collect(Collectors.toList());
  }
}
