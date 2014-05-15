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
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.checks.SquidCheck;

@Rule(
  key = "S1722",
  priority = Priority.MAJOR)
public class NewStyleClassCheck extends SquidCheck<Grammar> {

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
