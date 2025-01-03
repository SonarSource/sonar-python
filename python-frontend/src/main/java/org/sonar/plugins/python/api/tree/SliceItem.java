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
package org.sonar.plugins.python.api.tree;

import javax.annotation.CheckForNull;

/**
 * <pre>
 *   {@link #lowerBound()} {@link #boundSeparator()} {@link #upperBound()} {@link #strideSeparator()} {@link #stride()}
 * </pre>
 *
 * Examples:
 * <ul>
 *   <li><pre>1:10</pre></li>
 *   <li><pre>1:10:2</pre></li>
 *   <li><pre>1:</pre></li>
 *   <li><pre>1:</pre></li>
 *   <li><pre>:10</pre></li>
 *   <li><pre>:</pre></li>
 * </ul>
 * https://docs.python.org/3/reference/expressions.html#slicings
 */
public interface SliceItem extends Tree {

  @CheckForNull
  Expression lowerBound();

  Token boundSeparator();

  @CheckForNull
  Expression upperBound();

  @CheckForNull
  Token strideSeparator();

  @CheckForNull
  Expression stride();
}
