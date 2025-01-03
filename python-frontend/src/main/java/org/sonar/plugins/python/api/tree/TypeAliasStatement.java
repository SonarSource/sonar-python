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
 * <pre>
 *   type {@link #name()} {@link #typeParams()} = {@link #expression()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/simple_stmts.html#the-type-statement
 */
public interface TypeAliasStatement extends Statement {
  Token typeKeyword();

  Name name();

  @CheckForNull
  TypeParams typeParams();

  Token equalToken();

  Expression expression();

}
