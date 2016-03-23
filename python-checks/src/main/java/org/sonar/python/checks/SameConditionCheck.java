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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;
import org.sonar.sslr.ast.AstSelect;

import javax.annotation.Nullable;
import java.util.LinkedList;
import java.util.List;

@Rule(
    key = SameConditionCheck.CHECK_KEY,
    priority = Priority.CRITICAL,
    name = "Conditions in related \"if/elif/else if\" statements should not have the same condition",
    tags = {Tags.BUG, Tags.UNUSED, Tags.CERT, Tags.PITFALL}
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.LOGIC_RELIABILITY)
@SqaleConstantRemediation("10min")
@ActivatedByDefault
public class SameConditionCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "S1862";

  private List<AstNode> ignoreList;

  @Override
  public void init() {
    subscribeTo(PythonGrammar.IF_STMT);
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
        String message = String.format("This branch duplicates the one on line %s.", conditions.get(j).getToken().getLine());
        getContext().createLineViolation(this, message, conditions.get(index).getToken().getLine());
        return;
      }
    }
  }

  private AstNode singleIfChild(AstNode suite) {
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
