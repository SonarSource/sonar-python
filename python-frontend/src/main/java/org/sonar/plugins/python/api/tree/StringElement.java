/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

  boolean isTemplate();
  /**
   * @return Formatted expressions of an f-string or template string.
   * Empty list if the string element has no replacement fields.
   */
  List<FormattedExpression> formattedExpressions();
}
