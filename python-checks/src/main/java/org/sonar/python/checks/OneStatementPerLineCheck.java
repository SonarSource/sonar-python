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

import com.google.common.collect.Maps;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.Map;

/**
 * Note that implementation differs from AbstractOneStatementPerLineCheck due to Python specifics
 */
@Rule(
    key = OneStatementPerLineCheck.CHECK_KEY,
    priority = Priority.MINOR,
    name = "Statements should be on separate lines",
    tags = Tags.CONVENTION
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.READABILITY)
@SqaleConstantRemediation("1min")
@ActivatedByDefault
public class OneStatementPerLineCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "OneStatementPerLine";
  private final Map<Integer, Integer> statementsPerLine = Maps.newHashMap();

  @Override
  public void init() {
    subscribeTo(PythonGrammar.SIMPLE_STMT, PythonGrammar.SUITE);
  }

  @Override
  public void visitFile(AstNode astNode) {
    statementsPerLine.clear();
  }

  @Override
  public void visitNode(AstNode statementNode) {
    int line = statementNode.getTokenLine();
    if (!statementsPerLine.containsKey(line)) {
      statementsPerLine.put(line, 0);
    }
    statementsPerLine.put(line, statementsPerLine.get(line) + 1);
  }

  @Override
  public void leaveFile(AstNode astNode) {
    for (Map.Entry<Integer, Integer> statementsAtLine : statementsPerLine.entrySet()) {
      if (statementsAtLine.getValue() > 1) {
        getContext().createLineViolation(this, "At most one statement is allowed per line, but {0} statements were found on this line.", statementsAtLine.getKey(),
            statementsAtLine.getValue());
      }
    }
  }

}
