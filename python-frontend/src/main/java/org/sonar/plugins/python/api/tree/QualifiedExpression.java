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
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.python.types.v2.PythonType;

/**
 * Qualified expression like "foo.bar"
 *
 * <pre>
 *   {@link #qualifier()}.{@link #name()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/expressions.html#grammar-token-attributeref
 */
public interface QualifiedExpression extends Expression, HasSymbol {
  Expression qualifier();

  Token dotToken();

  Name name();

  /**
   * Returns the symbol of {@link #name()}
   */
  @CheckForNull
  @Override
  default Symbol symbol() {
    return name().symbol();
  }

  /**
   * Returns the usage of {@link #name()}
   */
  @CheckForNull
  @Override
  default Usage usage() {
    return name().usage();
  }

  @Override
  default PythonType typeV2() {
    return name().typeV2();
  }
}
