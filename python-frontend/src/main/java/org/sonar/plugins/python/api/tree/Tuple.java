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
package org.sonar.plugins.python.api.tree;

import java.util.List;
import javax.annotation.CheckForNull;

/**
 * <pre>
 *   ( {@link #elements()} )
 * </pre>
 *
 * or
 *
 * <pre>
 *   {@link #elements()}
 * </pre>
 *
 * Note that {@code ( a )} is not a tuple, but a parenthesized expression
 *
 *  Examples:
 *  <ul>
 *    <li><pre>(a, b)</pre></li>
 *    <li><pre>a, b</pre></li>
 *    <li><pre>a,</pre></li>
 *    <li><pre>(a,)</pre></li>
 *    <li>Empty tuple <pre>()</pre></li>
 *  </ul>
 *  See https://docs.python.org/3/library/stdtypes.html?highlight=tuple#tuple
 */
public interface Tuple extends Expression {

  @CheckForNull
  Token leftParenthesis();

  List<Expression> elements();

  List<Token> commas();

  @CheckForNull
  Token rightParenthesis();

}
