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

/**
 * <pre>
 *   lambda {@link FunctionLike#parameters()} : {@link #expression()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/expressions.html#grammar-token-lambda-expr
 */
public interface LambdaExpression extends Expression, FunctionLike {
  Token lambdaKeyword();

  Token colonToken();

  Expression expression();
}
