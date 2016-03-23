/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import com.google.common.base.Predicate;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;
import org.sonar.sslr.ast.AstSelect;

@Rule(
    key = EmptyNestedBlockCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Nested blocks of code should not be left empty",
    tags = Tags.BUG
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.LOGIC_RELIABILITY)
@SqaleConstantRemediation("5min")
@ActivatedByDefault
public class EmptyNestedBlockCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "S108";
  private static final Predicate<AstNode> NOT_PASS_PREDICATE = new NotPassPredicate();
  private static final String MESSAGE = "Either remove or fill this block of code.";

  @Override
  public void init() {
    subscribeTo(PythonGrammar.SUITE);
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
      getContext().createLineViolation(this, MESSAGE, stmtLists.get(0));
    }
  }

  private boolean isInExcept(AstNode suiteNode) {
    return suiteNode.getParent().is(PythonGrammar.TRY_STMT)
      && suiteNode.getPreviousSibling().getPreviousSibling().is(PythonGrammar.EXCEPT_CLAUSE);
  }

  private boolean containsComment(AstNode suiteNode) {
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
    public boolean apply(AstNode node) {
      return !node.getType().equals(PythonGrammar.PASS_STMT);
    }

  }

}
