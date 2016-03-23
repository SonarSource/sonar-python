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
import org.sonar.python.api.PythonPunctuator;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.List;

@Rule(
    key = PreIncrementDecrementCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "Pre-increment and pre-decrement should not be used",
    tags = Tags.BUG
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.INSTRUCTION_RELIABILITY)
@SqaleConstantRemediation("5min")
@ActivatedByDefault
public class PreIncrementDecrementCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "PreIncrementDecrement";

  @Override
  public void init() {
    subscribeTo(PythonGrammar.FACTOR);
  }

  @Override
  public void visitNode(AstNode astNode) {
    List<AstNode> children = astNode.getChildren();
    AstNode firstChild = children.get(0);
    AstNode secondChild = children.get(1);
    if (firstChild.is(PythonPunctuator.PLUS) && secondChild.getFirstChild().is(PythonPunctuator.PLUS)) {
      getContext().createLineViolation(this, "This statement doesn't produce the expected result, replace use of non-existent pre-increment operator", astNode);
    }
    if (firstChild.is(PythonPunctuator.MINUS) && secondChild.getFirstChild().is(PythonPunctuator.MINUS)) {
      getContext().createLineViolation(this, "This statement doesn't produce the expected result, replace use of non-existent pre-decrement operator", astNode);
    }
  }

}
