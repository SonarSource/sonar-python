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

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.BreakStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class BreakStatementImpl extends SimpleStatement implements BreakStatement {
  private final Token breakKeyword;
  private final Separators separators;

  public BreakStatementImpl(Token breakKeyword, Separators separators) {
    this.breakKeyword = breakKeyword;
    this.separators = separators;
  }

  @Override
  public Token breakKeyword() {
    return breakKeyword;
  }

  @CheckForNull
  @Override
  public Token separator() {
    return separators.last();
  }

  @Override
  public Kind getKind() {
    return Kind.BREAK_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitBreakStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(breakKeyword), separators.elements()).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
