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
package org.sonar.python.metrics;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.api.PythonMetric;
import org.sonar.squidbridge.SquidAstVisitor;

public class ComplexityVisitor extends SquidAstVisitor<Grammar> {

  @Override
  public void init() {
    subscribeTo(
      // Entry points
      PythonGrammar.FUNCDEF,

      // Branching nodes
      // Note that IF_STMT covered by PythonKeyword.IF below
      PythonGrammar.WHILE_STMT,
      PythonGrammar.FOR_STMT,
      PythonGrammar.RETURN_STMT,
      PythonGrammar.RAISE_STMT,
      PythonGrammar.EXCEPT_CLAUSE,

      // Expressions
      PythonKeyword.IF,
      PythonKeyword.AND,
      PythonKeyword.OR);
  }

  @Override
  public void visitNode(AstNode astNode) {
    getContext().peekSourceCode().add(PythonMetric.COMPLEXITY, 1);
  }

}
