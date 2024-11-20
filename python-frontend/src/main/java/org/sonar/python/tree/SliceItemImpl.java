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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.SliceItem;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class SliceItemImpl extends PyTree implements SliceItem {

  private final Expression lowerBound;
  private final Token boundSeparator;
  private final Expression upperBound;
  private final Token strideSeparator;
  private final Expression stride;

  public SliceItemImpl(
    @Nullable Expression lowerBound, Token boundSeparator, @Nullable Expression upperBound,
    @Nullable Token strideSeparator, @Nullable Expression stride
  ) {
    this.lowerBound = lowerBound;
    this.boundSeparator = boundSeparator;
    this.upperBound = upperBound;
    this.strideSeparator = strideSeparator;
    this.stride = stride;
  }

  @CheckForNull
  @Override
  public Expression lowerBound() {
    return lowerBound;
  }

  @Override
  public Token boundSeparator() {
    return boundSeparator;
  }

  @CheckForNull
  @Override
  public Expression upperBound() {
    return upperBound;
  }

  @CheckForNull
  @Override
  public Token strideSeparator() {
    return strideSeparator;
  }

  @CheckForNull
  @Override
  public Expression stride() {
    return stride;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitSliceItem(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(lowerBound, boundSeparator, upperBound, strideSeparator, stride).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return Kind.SLICE_ITEM;
  }
}
