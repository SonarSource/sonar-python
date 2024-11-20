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

import javax.annotation.CheckForNull;
import org.sonar.api.Beta;

/**
 * <pre>
 *   except {@link #exception()} as {@link #exceptionInstance()}:
 *     {@link #body()}
 * </pre>
 *
 * or (Python 2 syntax)
 * <pre>
 *   except {@link #exception()} , {@link #exceptionInstance()}:
 *     {@link #body()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/compound_stmts.html#the-try-statement
 */
public interface ExceptClause extends Tree {
  Token exceptKeyword();

  @CheckForNull
  Token starToken();

  @CheckForNull
  Expression exception();

  @CheckForNull
  Token asKeyword();

  @CheckForNull
  Token commaToken();

  @CheckForNull
  Expression exceptionInstance();
  
  @Beta
  Token colon();

  StatementList body();

}
