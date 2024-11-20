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

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.CaseBlock;
import org.sonar.plugins.python.api.tree.Guard;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class CaseBlockImpl extends PyTree implements CaseBlock {
  private final Token caseKeyword;
  private final Pattern pattern;
  @Nullable
  private final Guard guard;
  private final Token colon;
  @Nullable
  private final Token newLine;
  @Nullable
  private final Token indent;
  private final StatementList body;
  @Nullable
  private final Token dedent;

  public CaseBlockImpl(Token caseKeyword, Pattern pattern, @Nullable Guard guard, Token colon, @Nullable Token newLine,
    @Nullable Token indent, StatementList body, @Nullable Token dedent) {

    this.caseKeyword = caseKeyword;
    this.pattern = pattern;
    this.guard = guard;
    this.colon = colon;
    this.newLine = newLine;
    this.indent = indent;
    this.body = body;
    this.dedent = dedent;
  }


  @Override
  public Token caseKeyword() {
    return caseKeyword;
  }

  @Override
  public Pattern pattern() {
    return pattern;
  }

  @CheckForNull
  @Override
  public Guard guard() {
    return guard;
  }

  @Override
  public Token colon() {
    return colon;
  }

  @Override
  public StatementList body() {
    return body;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitCaseBlock(this);
  }

  @Override
  public Kind getKind() {
    return Kind.CASE_BLOCK;
  }

  @Override
  List<Tree> computeChildren() {
    return Stream.of(caseKeyword, pattern, guard, colon, newLine, indent, body, dedent)
      .filter(Objects::nonNull)
      .toList();
  }
}
