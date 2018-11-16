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
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.sslr.ast.AstSelect;

@Rule(key = SameConditionCheck.CHECK_KEY)
public class SameConditionCheck extends PythonCheck {
  public static final String CHECK_KEY = "S1862";
  private static final String MESSAGE = "This branch duplicates the one on line %s.";

  private List<AstNode> ignoreList;

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.IF_STMT);
  }

  @Override
  public void visitFile(@Nullable AstNode astNode) {
    ignoreList = new LinkedList<>();
  }

  @Override
  public void visitNode(AstNode node) {
    if (ignoreList.contains(node)) {
      return;
    }
    List<AstNode> conditions = getConditionsToCompare(node);
    findSameConditions(conditions);
  }

  private List<AstNode> getConditionsToCompare(AstNode ifStmt) {
    List<AstNode> conditions = ifStmt.getChildren(PythonGrammar.TEST);
    AstNode elseNode = ifStmt.getFirstChild(PythonKeyword.ELSE);
    if (conditions.size() == 1 && elseNode != null) {
      AstNode suite = elseNode.getNextSibling().getNextSibling();
      lookForElseIfs(conditions, suite);
    }
    return conditions;
  }

  private void lookForElseIfs(List<AstNode> conditions, AstNode suite) {
    AstNode singleIfChild = singleIfChild(suite);
    if (singleIfChild != null) {
      ignoreList.add(singleIfChild);
      conditions.addAll(getConditionsToCompare(singleIfChild));
    }
  }

  private void findSameConditions(List<AstNode> conditions) {
    for (int i = 1; i < conditions.size(); i++) {
      checkCondition(conditions, i);
    }
  }

  private void checkCondition(List<AstNode> conditions, int index) {
    for (int j = 0; j < index; j++) {
      if (CheckUtils.equalNodes(conditions.get(j), conditions.get(index))) {
        String message = String.format(MESSAGE, conditions.get(j).getToken().getLine());
        addIssue(conditions.get(index), message).secondary(conditions.get(j), "Original");
        return;
      }
    }
  }

  private static AstNode singleIfChild(AstNode suite) {
    List<AstNode> statements = suite.getChildren(PythonGrammar.STATEMENT);
    if (statements.size() == 1) {
      AstSelect nestedIf = statements.get(0).select()
          .children(PythonGrammar.COMPOUND_STMT)
          .children(PythonGrammar.IF_STMT);
      if (nestedIf.size() == 1) {
        return nestedIf.get(0);
      }
    }
    return null;
  }
}
