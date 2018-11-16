/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import java.util.Collections;
import java.util.Set;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.sslr.ast.AstSelect;

@Rule(key = EmptyNestedBlockCheck.CHECK_KEY)
public class EmptyNestedBlockCheck extends PythonCheck {
  public static final String CHECK_KEY = "S108";
  private static final Predicate<AstNode> NOT_PASS_PREDICATE = new NotPassPredicate();
  private static final String MESSAGE = "Either remove or fill this block of code.";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.SUITE);
  }

  @Override
  public void visitNode(AstNode suiteNode) {
    if (suiteNode.getParent().is(PythonGrammar.FUNCDEF, PythonGrammar.CLASSDEF) || isInExcept(suiteNode)) {
      return;
    }

    AstSelect suite = suiteNode.select();
    AstSelect stmtLists = suite.children(PythonGrammar.STMT_LIST);
    if (stmtLists.isEmpty()) {
      AstSelect statementSelect = suite.children(PythonGrammar.STATEMENT);
      if (statementSelect.children(PythonGrammar.COMPOUND_STMT).isNotEmpty()) {
        return;
      }
      stmtLists = statementSelect.children(PythonGrammar.STMT_LIST);
    }

    AstSelect nonPassSimpleStatements = stmtLists
      .children(PythonGrammar.SIMPLE_STMT)
      .children()
      .filter(NOT_PASS_PREDICATE);
    if (nonPassSimpleStatements.isEmpty() && !containsComment(suiteNode)) {
      addIssue(stmtLists.get(0), MESSAGE);
    }
  }

  private static boolean isInExcept(AstNode suiteNode) {
    return suiteNode.getParent().is(PythonGrammar.TRY_STMT)
      && suiteNode.getPreviousSibling().getPreviousSibling().is(PythonGrammar.EXCEPT_CLAUSE);
  }

  private static boolean containsComment(AstNode suiteNode) {
    for (Token token : suiteNode.getTokens()) {
      for (Trivia trivia : token.getTrivia()) {
        if (trivia.isComment()) {
          return true;
        }
      }
    }
    return false;
  }

  private static class NotPassPredicate implements Predicate<AstNode> {

    @Override
    public boolean test(AstNode node) {
      return !node.getType().equals(PythonGrammar.PASS_STMT);
    }

  }

}
