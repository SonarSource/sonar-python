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
import org.sonar.plugins.python.api.tree.Token;

public class Separators {
  public static final Separators EMPTY = new Separators(null, null);
  @Nullable
  private final Token separator;
  @Nullable
  private final Token newline;
  private final List<Token> elements;

  Separators(@Nullable Token separator, @Nullable Token newline) {
    this.separator = separator;
    this.newline = newline;
    this.elements = Stream.of(this.separator, this.newline).filter(Objects::nonNull).toList();
  }

  @CheckForNull
  public Token last() {
    return newline == null ? separator : newline;
  }

  public List<Token> elements() {
    return elements;
  }
}
