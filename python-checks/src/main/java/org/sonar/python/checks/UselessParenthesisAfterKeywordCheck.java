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
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;

@Rule(key = UselessParenthesisAfterKeywordCheck.CHECK_KEY)
public class UselessParenthesisAfterKeywordCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1721";
  private static final Map<PythonGrammar, String> KEYWORDS_FOLLOWED_BY_TEST = initializeKeywordsFollowedByTest();

  private static Map<PythonGrammar, String> initializeKeywordsFollowedByTest() {
    Map<PythonGrammar, String> map = new EnumMap<>(PythonGrammar.class);
    map.put(PythonGrammar.ASSERT_STMT, "assert");
    map.put(PythonGrammar.RAISE_STMT, "raise");
    map.put(PythonGrammar.WHILE_STMT, "while");
    return Collections.unmodifiableMap(map);
  }

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(
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
    String keyword = KEYWORDS_FOLLOWED_BY_TEST.get(node.getType());
    if (keyword != null) {
      checkParenthesis(node.getFirstChild(PythonGrammar.TEST), keyword, node);
    } else if (node.is(PythonGrammar.DEL_STMT)) {
      checkParenthesis(node.getFirstChild(PythonGrammar.EXPRLIST), "del", node);
    } else if (node.is(PythonGrammar.IF_STMT)) {
      List<AstNode> testNodes = node.getChildren(PythonGrammar.TEST);
      checkParenthesis(testNodes.get(0), "if", node);
      if (testNodes.size() > 1) {
        checkParenthesis(testNodes.get(1), "elif", testNodes.get(1));
      }
    } else if (node.is(PythonGrammar.FOR_STMT)) {
      visitForExpression(node);
      checkParenthesis(node.getFirstChild(PythonGrammar.TESTLIST), "in", node);
    } else if (node.is(PythonGrammar.RETURN_STMT)) {
      checkParenthesis(node.getFirstChild(PythonGrammar.TESTLIST), "return", node);
    } else if (node.is(PythonGrammar.YIELD_EXPR)) {
      checkParenthesis(node.getFirstChild(PythonGrammar.TESTLIST), "yield", node);
    } else if (node.is(PythonGrammar.EXCEPT_CLAUSE)) {
      visitExceptClause(node);
    } else if (node.is(PythonGrammar.NOT_TEST)) {
      visitNotTest(node);
    }
  }

  private void visitForExpression(AstNode node) {
    if (node.getFirstChild(PythonGrammar.EXPRLIST).getNumberOfChildren() == 1) {
      checkParenthesis(node.getFirstChild(PythonGrammar.EXPRLIST), "for", node);
    }
  }

  private void visitNotTest(AstNode node) {
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

  private void visitExceptClause(AstNode node) {
    int nbTests = node.select()
      .children(PythonGrammar.TEST)
      .children(PythonGrammar.ATOM)
      .children(PythonGrammar.TESTLIST_COMP)
      .children(PythonGrammar.TEST)
      .size();
    if (nbTests == 1) {
      checkParenthesis(node.getFirstChild(PythonGrammar.TEST), "except", node);
    }
  }

  private void checkParenthesis(@Nullable AstNode child, String keyword, AstNode errorNode) {
    if (child != null && child.getToken().getType().equals(PythonPunctuator.LPARENTHESIS) && isOnASingleLine(child)) {
      String message = String.format("Remove the parentheses after this \"%s\" keyword.", keyword);
      addLineIssue(message, errorNode.getTokenLine());
    }
  }

  private static boolean isOnASingleLine(AstNode node) {
    return node.getTokenLine() == node.getLastToken().getLine();
  }

}
