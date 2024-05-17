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
  default PythonType pythonType() {
    return name().pythonType();
  }
}
