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
package org.sonar.python.api.tree;

import com.sonar.sslr.api.AstNode;

public interface Tree {

  void accept(PyTreeVisitor visitor);

  boolean is(Kind kind);

  AstNode astNode();

  enum Kind {
    ASSERT_STMT(PyAssertStatementTree.class),

    BREAK_STMT(PyBreakStatementTree.class),

    CLASSDEF(PyClassDefTree.class),

    CONTINUE_STMT(PyContinueStatementTree.class),

    DEL_STMT(PyDelStatementTree.class),

    ELSE_STMT(PyElseStatementTree.class),

    EXEC_STMT(PyExecStatementTree.class),

    FILE_INPUT(PyFileInputTree.class),

    FUNCDEF(PyFunctionDefTree.class),

    IF_STMT(PyIfStatementTree.class),

    NAME(PyNameTree.class),

    PASS_STMT(PyPassStatementTree.class),

    PRINT_STMT(PyPrintStatementTree.class),

    RAISE_STMT(PyRaiseStatementTree.class),

    RETURN_STMT(PyReturnStatementTree.class),

    YIELD_EXPR(PyYieldExpressionTree.class),

    YIELD_STMT(PyYieldStatementTree.class);

    final Class<? extends Tree> associatedInterface;

    Kind(Class<? extends Tree> associatedInterface) {
      this.associatedInterface = associatedInterface;
    }
  }
}
