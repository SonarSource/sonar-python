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
package org.sonar.plugins.python.api.tree;

/**
 * <pre>
 * {@link #starToken()} {@link #expression()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/expressions.html#grammar-token-starred-expression
 * See https://www.python.org/dev/peps/pep-0448/
 */
public interface UnpackingExpression extends Expression, DictionaryLiteralElement, Argument {
  /**
   * Can be either {@code *} or {@code **}
   */
  Token starToken();

  Expression expression();

}
