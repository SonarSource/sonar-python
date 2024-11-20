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

/**
 * <pre>
 *   {@link #trueExpression()} if {@link #condition()} else {@link #falseExpression()}
 * </pre>
 *
 * Example: <pre>1 if x == 42 else 2</pre>
 *
 * See https://docs.python.org/3/reference/expressions.html#conditional-expressions
 */
public interface ConditionalExpression extends Expression {
  Expression trueExpression();

  Token ifKeyword();

  Expression condition();

  Token elseKeyword();

  Expression falseExpression();
}
