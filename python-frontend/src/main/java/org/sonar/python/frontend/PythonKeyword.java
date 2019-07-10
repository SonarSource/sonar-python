/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.frontend;

import com.jetbrains.python.PyTokenTypes;
import com.jetbrains.python.psi.PyElementType;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class PythonKeyword {

  private PythonKeyword() {
    // empty constructor
  }

  private static final Set<PyElementType> KEYWORDS_ELEMENT_TYPES = new HashSet<>(Arrays.asList(
    PyTokenTypes.AND_KEYWORD,
    PyTokenTypes.AS_KEYWORD,
    PyTokenTypes.ASSERT_KEYWORD,
    PyTokenTypes.BREAK_KEYWORD,
    PyTokenTypes.CLASS_KEYWORD,
    PyTokenTypes.CONTINUE_KEYWORD,
    PyTokenTypes.DEF_KEYWORD,
    PyTokenTypes.DEL_KEYWORD,
    PyTokenTypes.ELIF_KEYWORD,
    PyTokenTypes.ELSE_KEYWORD,
    PyTokenTypes.EXCEPT_KEYWORD,
    PyTokenTypes.EXEC_KEYWORD,
    PyTokenTypes.FINALLY_KEYWORD,
    PyTokenTypes.FOR_KEYWORD,
    PyTokenTypes.FROM_KEYWORD,
    PyTokenTypes.GLOBAL_KEYWORD,
    PyTokenTypes.IF_KEYWORD,
    PyTokenTypes.IMPORT_KEYWORD,
    PyTokenTypes.IN_KEYWORD,
    PyTokenTypes.IS_KEYWORD,
    PyTokenTypes.LAMBDA_KEYWORD,
    PyTokenTypes.NOT_KEYWORD,
    PyTokenTypes.OR_KEYWORD,
    PyTokenTypes.PASS_KEYWORD,
    PyTokenTypes.PRINT_KEYWORD,
    PyTokenTypes.RAISE_KEYWORD,
    PyTokenTypes.RETURN_KEYWORD,
    PyTokenTypes.TRY_KEYWORD,
    PyTokenTypes.WITH_KEYWORD,
    PyTokenTypes.WHILE_KEYWORD,
    PyTokenTypes.YIELD_KEYWORD,
    PyTokenTypes.NONE_KEYWORD,
    PyTokenTypes.TRUE_KEYWORD,
    PyTokenTypes.FALSE_KEYWORD,
    PyTokenTypes.NONLOCAL_KEYWORD,
    PyTokenTypes.DEBUG_KEYWORD,
    PyTokenTypes.ASYNC_KEYWORD,
    PyTokenTypes.AWAIT_KEYWORD));

  public static boolean isKeyword(PyElementType elementType) {
    return KEYWORDS_ELEMENT_TYPES.contains(elementType);
  }
}
