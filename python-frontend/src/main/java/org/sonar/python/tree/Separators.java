/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
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

  Separators(@Nullable AstNode separator, @Nullable AstNode newline){
    this.separator = separator == null ? null : new TokenImpl(separator.getToken());
    this.newline = newline == null ? null : new TokenImpl(newline.getToken());
    this.elements = Stream.of(this.separator, this.newline).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @CheckForNull
  public Token last() {
    return newline == null ? separator : newline;
  }

  public List<Token> elements() {
    return elements;
  }
}
