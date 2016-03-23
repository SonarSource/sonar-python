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
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
    key = NewStyleClassCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "New-style classes should be used"
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.INSTRUCTION_RELIABILITY)
@SqaleConstantRemediation("2min")
public class NewStyleClassCheck extends SquidCheck<Grammar> {

  public static final String CHECK_KEY = "S1722";

  @Override
  public void init() {
    subscribeTo(PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode argListNode = node.getFirstChild(PythonGrammar.ARGLIST);
    if (argListNode == null || !argListNode.hasDirectChildren(PythonGrammar.ARGUMENT)) {
      getContext().createLineViolation(this,
        "Add inheritance from \"object\" or some other new-style class.", node.getFirstChild(PythonGrammar.CLASSNAME));
    }
  }

}
