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
