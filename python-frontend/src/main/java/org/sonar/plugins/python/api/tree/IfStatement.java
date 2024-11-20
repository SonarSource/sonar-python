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

import java.util.List;
import javax.annotation.CheckForNull;

/**
 * if-elif-else statement.
 *
 * Note that this interface has a recursive structure because it represents both 'if' clause and 'elif' clause
 *
 * <pre>
 *   if {@link #condition()}:
 *     {@link #body()}
 *   {@link #elseBranch()}
 * </pre>
 *
 * or
 *
 * <pre>
 *   elif {@link #condition()}:
 *     {@link #body()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/compound_stmts.html#grammar-token-if-stmt
 */
public interface IfStatement extends Statement {
  Token keyword();

  Expression condition();

  StatementList body();

  List<IfStatement> elifBranches();

  boolean isElif();

  @CheckForNull
  ElseClause elseBranch();

}
