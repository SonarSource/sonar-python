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
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.PassStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class PassStatementImpl extends SimpleStatement implements PassStatement {
  private final Token passKeyword;
  private final Separators separators;

  public PassStatementImpl(Token passKeyword, Separators separators) {
    this.passKeyword = passKeyword;
    this.separators = separators;
  }

  @Override
  public Token passKeyword() {
    return passKeyword;
  }

  @Nullable
  @Override
  public Token separator() {
    return separators.last();
  }

  @Override
  public Kind getKind() {
    return Kind.PASS_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitPassStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(passKeyword), separators.elements()).flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
