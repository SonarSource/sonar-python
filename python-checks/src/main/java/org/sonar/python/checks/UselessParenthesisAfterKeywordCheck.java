/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.check.BelongsToProfile;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.List;

@Rule(
  key = "S1721",
  priority = Priority.MAJOR)
@BelongsToProfile(title = CheckList.SONAR_WAY_PROFILE, priority = Priority.MAJOR)
public class UselessParenthesisAfterKeywordCheck extends SquidCheck<Grammar> {

  @Override
  public void init() {
    subscribeTo(
      PythonGrammar.ASSERT_STMT,
      PythonGrammar.DEL_STMT,
      PythonGrammar.IF_STMT,
      PythonGrammar.FOR_STMT,
      PythonGrammar.RAISE_STMT,
      PythonGrammar.RETURN_STMT,
      PythonGrammar.WHILE_STMT,
      PythonGrammar.YIELD_EXPR,
      PythonGrammar.EXCEPT_CLAUSE,
      PythonGrammar.NOT_TEST);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode firstTestChild = node.getFirstChild(PythonGrammar.TEST);
    if (node.is(PythonGrammar.ASSERT_STMT)) {
      checkParenthesis(firstTestChild, "assert", node);
    } else if (node.is(PythonGrammar.DEL_STMT)) {
      checkParenthesis(node.getFirstChild(PythonGrammar.EXPRLIST), "del", node);
    } else if (node.is(PythonGrammar.IF_STMT)) {
      List<AstNode> testNodes = node.getChildren(PythonGrammar.TEST);
      checkParenthesis(testNodes.get(0), "if", node);
      if (testNodes.size() > 1) {
        checkParenthesis(testNodes.get(1), "elif", testNodes.get(1));
      }
    } else if (node.is(PythonGrammar.FOR_STMT)) {
      checkParenthesis(node.getFirstChild(PythonGrammar.EXPRLIST), "for", node);
      checkParenthesis(node.getFirstChild(PythonGrammar.TESTLIST), "in", node);
    } else if (node.is(PythonGrammar.RAISE_STMT)) {
      checkParenthesis(firstTestChild, "raise", node);
    } else if (node.is(PythonGrammar.RETURN_STMT)) {
      checkParenthesis(node.getFirstChild(PythonGrammar.TESTLIST), "return", node);
    } else if (node.is(PythonGrammar.WHILE_STMT)) {
      checkParenthesis(firstTestChild, "while", node);
    } else if (node.is(PythonGrammar.YIELD_EXPR)) {
      checkParenthesis(node.getFirstChild(PythonGrammar.TESTLIST), "yield", node);
    } else if (node.is(PythonGrammar.EXCEPT_CLAUSE)) {
      int nbTests = node.select()
        .children(PythonGrammar.TEST)
        .children(PythonGrammar.ATOM)
        .children(PythonGrammar.TESTLIST_COMP)
        .children(PythonGrammar.TEST)
        .size();
      if (nbTests == 1) {
        checkParenthesis(firstTestChild, "except", node);
      }
    } else if (node.is(PythonGrammar.NOT_TEST)) {
      boolean hasUselessParenthesis = node.select()
        .children(PythonGrammar.ATOM)
        .children(PythonGrammar.TESTLIST_COMP)
        .children(PythonGrammar.TEST)
        .children(PythonGrammar.ATOM, PythonGrammar.COMPARISON)
        .isNotEmpty();
      if (hasUselessParenthesis) {
        checkParenthesis(node.getFirstChild().getNextSibling(), "not", node);
      }
    }
  }

  private void checkParenthesis(AstNode child, String keyword, AstNode errorNode) {
    if (child != null && child.getToken().getType() == PythonPunctuator.LPARENTHESIS) {
      getContext().createLineViolation(this,
        "Remove the parentheses after this \"{0}\"", errorNode, keyword);
    }
  }

}
