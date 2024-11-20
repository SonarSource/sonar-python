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
