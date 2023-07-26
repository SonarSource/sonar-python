/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import com.google.common.annotations.Beta;
import java.util.List;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

/**
 * <pre>
 *   return {@link #expressions()}
 * </pre>
 *
 * See https://docs.python.org/3/reference/simple_stmts.html#grammar-token-return-stmt
 */
public interface ReturnStatement extends Statement {
  Token returnKeyword();

  List<Expression> expressions();

  @Beta
  List<Token> commas();

  /**
   * Infers the type of the returned value that would result from the execution of this return statement.
   */
  @Beta
  default InferredType returnValueType() {
    var returnedExpressions = expressions();

    if (returnedExpressions.isEmpty()) {
      return InferredTypes.NONE;
    }

    if (returnedExpressions.size() == 1) {
      return returnedExpressions.get(0).type();
    }

    return InferredTypes.TUPLE;
  }
}
