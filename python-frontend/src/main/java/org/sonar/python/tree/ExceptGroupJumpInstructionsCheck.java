/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.tree;

import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BreakStatement;
import org.sonar.plugins.python.api.tree.ContinueStatement;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;

/**
 * Purpose of this class is to detect and throw exception if the scanned tree that contain invalid syntax regarding the except* instruction.
 * It is currently not valid to have a break, continue or return statement in an except* body.
 */
public class ExceptGroupJumpInstructionsCheck extends BaseTreeVisitor {
  @Override
  public void visitBreakStatement(BreakStatement breakStatement) {
    checkExceptGroupStatement(breakStatement, "break statement cannot appear in except* block",
      Tree.Kind.EXCEPT_GROUP_CLAUSE, Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT);
  }

  @Override
  public void visitContinueStatement(ContinueStatement continueStatement) {
    checkExceptGroupStatement(continueStatement, "continue statement cannot appear in except* block",
      Tree.Kind.EXCEPT_GROUP_CLAUSE, Tree.Kind.FOR_STMT, Tree.Kind.WHILE_STMT);
  }

  @Override
  public void visitReturnStatement(ReturnStatement returnStatement) {
    checkExceptGroupStatement(returnStatement, "return statement cannot appear in except* block",
      Tree.Kind.EXCEPT_GROUP_CLAUSE, Tree.Kind.FUNCDEF);
  }

  private static void checkExceptGroupStatement(Tree statement, String message, Tree.Kind... possibleParents) {
    Tree parent = TreeUtils.firstAncestorOfKind(statement, possibleParents);
    if (parent != null && parent.is(Tree.Kind.EXCEPT_GROUP_CLAUSE)) {
      PythonTreeMaker.recognitionException(statement.firstToken().line(), message);
    }
  }
}
