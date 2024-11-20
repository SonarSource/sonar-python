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

import java.util.Objects;
import org.sonar.plugins.python.api.tree.Token;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.Tree;

public class IfStatementImpl extends PyTree implements IfStatement {

  private final Token keyword;
  private final Expression condition;
  private final Token colon;
  @CheckForNull
  private final Token newLine;
  @CheckForNull
  private final Token indent;
  private final StatementList statements;
  private final Token dedent;
  private final List<IfStatement> elifBranches;
  private final boolean isElif;
  @CheckForNull
  private final ElseClause elseClause;

  /**
   * If statement constructor
   */
  public IfStatementImpl(Token ifKeyword, Expression condition,
                               Token colon, @CheckForNull Token newLine, @CheckForNull Token indent, StatementList statements, @CheckForNull Token dedent,
                               List<IfStatement> elifBranches, @CheckForNull ElseClause elseClause) {
    this.keyword = ifKeyword;
    this.condition = condition;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.statements = statements;
    this.dedent = dedent;
    this.elifBranches = elifBranches;
    this.isElif = false;
    this.elseClause = elseClause;
  }

  /**
   * Elif statement constructor
   */
  public IfStatementImpl(Token elifKeyword, Expression condition, Token colon,
                               @CheckForNull Token newLine, @CheckForNull Token indent, StatementList statements, @CheckForNull Token dedent) {
    this.keyword = elifKeyword;
    this.condition = condition;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.statements = statements;
    this.dedent = dedent;
    this.elifBranches = Collections.emptyList();
    this.isElif = true;
    this.elseClause = null;
  }

  @Override
  public Token keyword() {
    return keyword;
  }

  @Override
  public Expression condition() {
    return condition;
  }

  @Override
  public StatementList body() {
    return statements;
  }

  @Override
  public List<IfStatement> elifBranches() {
    return elifBranches;
  }

  @Override
  public boolean isElif() {
    return isElif;
  }

  @CheckForNull
  @Override
  public ElseClause elseBranch() {
    return elseClause;
  }

  @Override
  public Kind getKind() {
    return Tree.Kind.IF_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitIfStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Arrays.asList(keyword, condition, colon, newLine, indent, statements, dedent), elifBranches, Collections.singletonList(elseClause))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
