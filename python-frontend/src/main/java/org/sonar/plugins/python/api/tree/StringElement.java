/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.plugins.python.api.tree;

import java.util.List;

/**
 * <pre>
 *   {@link #prefix()}{@link #value()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/lexical_analysis.html#grammar-token-stringliteral
 */
public interface StringElement extends Tree {
  /**
   * @return the token value of this literal.
   */
  String value();

  String trimmedQuotesValue();

  String prefix();

  boolean isTripleQuoted();

  boolean isInterpolated();

  /**
   * @deprecated Use {@link #formattedExpressions()} instead.
   */
  @Deprecated
  List<Expression> interpolatedExpressions();

  /**
   * @return Formatted expressions of an f-string.
   * Empty list if the string element is not an f-string.
   */
  List<FormattedExpression> formattedExpressions();
}
