/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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

import java.util.Set;

import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.sslr.ast.AstSelect;

import com.google.common.collect.ImmutableSet;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;

@Rule(key = SelfAssignmentCheck.CHECK_KEY)
public class SelfAssignmentCheck extends PythonCheck {

  public static final String CHECK_KEY = "S1656";

  public static final String MESSAGE = "Remove or correct this useless self-assignment.";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return ImmutableSet.of(PythonGrammar.EXPRESSION_STMT);
  }

  @Override
  public void visitNode(AstNode node) {
    for (AstNode assignOperator : node.getChildren(PythonPunctuator.ASSIGN)) {
      if (CheckUtils.equalNodes(assignOperator.getPreviousSibling(), assignOperator.getNextSibling()) && !isException(node)) {
        addIssue(assignOperator, MESSAGE);
      }
    }
  }

  private static boolean isException(AstNode expressionStatement) {
    AstSelect potentialFunctionCalls = expressionStatement.select()
      .descendants(PythonGrammar.TRAILER)
      .children(PythonPunctuator.LPARENTHESIS);
    if (!potentialFunctionCalls.isEmpty()) {
      return true;
    }
    AstNode suite = expressionStatement.getFirstAncestor(PythonGrammar.SUITE);
    return suite != null && suite.getParent().is(PythonGrammar.CLASSDEF, PythonGrammar.TRY_STMT);
  }
}
