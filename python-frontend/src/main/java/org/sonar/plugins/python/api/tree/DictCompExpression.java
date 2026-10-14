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

import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.Symbol;

/**
 * Dictionary Comprehension
 * <pre>
 *  { {@link #resultElement()} {@link #comprehensionFor()} }
 * </pre>
 *
 * Examples: <pre> { key:value for i in items }
 * { **mapping for mapping in mappings } </pre>
 *
 * See https://docs.python.org/3/reference/expressions.html#grammar-token-dict-comprehension
 */
public interface DictCompExpression extends Expression {

  /**
   * The non-null comprehension head: either a {@link KeyValuePair}, or an
   * {@link UnpackingExpression} whose star token is {@code **}.
   * The element's parent is this comprehension. For a key/value head, the key,
   * colon and value are children of the {@code KeyValuePair}.
   */
  DictionaryLiteralElement resultElement();

  /**
   * Returns null for a mapping-unpacking comprehension.
   * @deprecated Use {@link #resultElement()} and {@link KeyValuePair#key()}.
   */
  @Deprecated(since = "5.33")
  @CheckForNull
  Expression keyExpression();

  /**
   * Returns null for a mapping-unpacking comprehension.
   * @deprecated Use {@link #resultElement()} and {@link KeyValuePair#colon()}.
   */
  @Deprecated(since = "5.33")
  @CheckForNull
  Token colonToken();

  /**
   * Returns null for a mapping-unpacking comprehension.
   * @deprecated Use {@link #resultElement()} and {@link KeyValuePair#value()}.
   */
  @Deprecated(since = "5.33")
  @CheckForNull
  Expression valueExpression();

  ComprehensionFor comprehensionFor();

  Set<Symbol> localVariables();
}
