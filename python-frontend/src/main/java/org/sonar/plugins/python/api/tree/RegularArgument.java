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

/**
 * <pre>{@link #keywordArgument()} = {@link #expression()}</pre> or <pre>{@link #expression()}</pre>
 *
 * See https://docs.python.org/3/reference/expressions.html#grammar-token-argument-list
 */
public interface RegularArgument extends Argument {
  @CheckForNull
  Name keywordArgument();

  @CheckForNull
  Token equalToken();

  Expression expression();
}
