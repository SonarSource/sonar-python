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
 * Dictionary Comprehension
 * <pre>
 *  { {@link #keyExpression()} : {@link #valueExpression()} {@link #comprehensionFor()} }
 * </pre>
 *
 * Example: <pre> { key:value for i in list } </pre>
 *
 * See https://docs.python.org/3/reference/expressions.html#grammar-token-dict-comprehension
 */
public interface DictCompExpression extends Expression {

  Expression keyExpression();

  Token colonToken();

  Expression valueExpression();

  ComprehensionFor comprehensionFor();

  Set<Symbol> localVariables();
}
