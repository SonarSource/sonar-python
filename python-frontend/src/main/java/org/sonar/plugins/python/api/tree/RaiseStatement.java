/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.List;
import javax.annotation.CheckForNull;

/**
 * <pre>
 *   raise {@link #expressions()} from {@link #fromExpression()}
 * </pre>
 * Example: <pre>raise foo from bar</pre>
 *
 * See https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement
 */
public interface RaiseStatement extends Statement {
  Token raiseKeyword();

  List<Expression> expressions();

  @CheckForNull
  Token fromKeyword();

  @CheckForNull
  Expression fromExpression();
}
