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

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NonlocalStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class NonlocalStatementImpl extends SimpleStatement implements NonlocalStatement {
  private final Token nonlocalKeyword;
  private final List<Name> variables;
  private final Separators separators;

  public NonlocalStatementImpl(Token nonlocalKeyword, List<Name> variables, Separators separators) {
    this.nonlocalKeyword = nonlocalKeyword;
    this.variables = variables;
    this.separators = separators;
  }

  @Override
  public Token nonlocalKeyword() {
    return nonlocalKeyword;
  }

  @Override
  public List<Name> variables() {
    return variables;
  }

  @CheckForNull
  @Override
  public Token separator() {
    return separators.last();
  }

  @Override
  public Kind getKind() {
    return Kind.NONLOCAL_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitNonlocalStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(nonlocalKeyword), variables, separators.elements())
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
