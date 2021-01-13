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

import java.util.Set;
import org.sonar.plugins.python.api.symbols.Symbol;

/**
 *
 * <pre>
 *   {@link #resultExpression()} {@link #comprehensionFor()}
 * </pre>
 *
 * Common interface for:
 *  <ul>
 *    <li>Set Comprehension <pre>{x for x in range(1, 100)}</pre></li>
 *    <li>List Comprehension <pre>[x for x in range(1, 100)]</pre></li>
 *    <li>Generator Expression <pre>x for x in range(1, 100)</pre></li>
 *  </ul>
 *
 *  See https://docs.python.org/3/reference/expressions.html#displays-for-lists-sets-and-dictionaries
 */
public interface ComprehensionExpression extends Expression {

  Expression resultExpression();

  ComprehensionFor comprehensionFor();

  /**
   * local variables are following python3 scoping rules for comprehension expressions.
   */
  Set<Symbol> localVariables();

}
