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
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;

@Rule(key = NewStyleClassCheck.CHECK_KEY)
public class NewStyleClassCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1722";
  private static final String MESSAGE = "Add inheritance from \"object\" or some other new-style class.";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return immutableSet(PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    AstNode argListNode = node.getFirstChild(PythonGrammar.ARGLIST);
    if (argListNode == null || !argListNode.hasDirectChildren(PythonGrammar.ARGUMENT)) {
      addIssue(node.getFirstChild(PythonGrammar.CLASSNAME), MESSAGE);
    }
  }

}
