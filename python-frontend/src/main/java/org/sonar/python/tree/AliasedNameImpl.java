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

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class AliasedNameImpl extends PyTree implements AliasedName {

  private final Token asKeyword;
  private final DottedName dottedName;
  private final Name alias;

  public AliasedNameImpl(Token asKeyword, DottedName dottedName, Name alias) {
    this.asKeyword = asKeyword;
    this.dottedName = dottedName;
    this.alias = alias;
  }

  public AliasedNameImpl(DottedName dottedName) {
    this.asKeyword = null;
    this.dottedName = dottedName;
    this.alias = null;
  }

  @CheckForNull
  @Override
  public Token asKeyword() {
    return asKeyword;
  }

  @CheckForNull
  @Override
  public Name alias() {
    return alias;
  }

  @Override
  public DottedName dottedName() {
    return dottedName;
  }

  @Override
  public Kind getKind() {
    return Kind.ALIASED_NAME;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitAliasedName(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(dottedName, asKeyword, alias).filter(Objects::nonNull).toList();
  }
}
