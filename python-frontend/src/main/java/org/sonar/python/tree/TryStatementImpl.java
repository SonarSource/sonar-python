/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.ElseClause;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.FinallyClause;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.TryStatement;

public class TryStatementImpl extends PyTree implements TryStatement {
  private final Token tryKeyword;
  private final Token colon;
  private final Token newLine;
  private final Token indent;
  private final StatementList tryBody;
  private final Token dedent;
  private final List<ExceptClause> exceptClauses;
  private final FinallyClause finallyClause;
  private final ElseClause elseClause;

  public TryStatementImpl(Token tryKeyword, Token colon, @Nullable Token newLine, @Nullable Token indent, StatementList tryBody,
                          @Nullable Token dedent, List<ExceptClause> exceptClauses, @Nullable FinallyClause finallyClause, @Nullable ElseClause elseClause) {
    this.tryKeyword = tryKeyword;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.tryBody = tryBody;
    this.dedent = dedent;
    this.exceptClauses = exceptClauses;
    this.finallyClause = finallyClause;
    this.elseClause = elseClause;
  }

  @Override
  public Token tryKeyword() {
    return tryKeyword;
  }

  @Override
  public List<ExceptClause> exceptClauses() {
    return exceptClauses;
  }

  @CheckForNull
  @Override
  public FinallyClause finallyClause() {
    return finallyClause;
  }

  @CheckForNull
  @Override
  public ElseClause elseClause() {
    return elseClause;
  }

  @Override
  public StatementList body() {
    return tryBody;
  }

  @Override
  public Kind getKind() {
    return Kind.TRY_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitTryStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Arrays.asList(tryKeyword, colon, newLine, indent, tryBody, dedent), exceptClauses, Arrays.asList(elseClause, finallyClause))
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
