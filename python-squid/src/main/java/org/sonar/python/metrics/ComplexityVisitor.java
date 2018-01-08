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
package org.sonar.python.metrics;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.sonar.python.PythonVisitor;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonKeyword;

public class ComplexityVisitor extends PythonVisitor {

  private int complexity;

  public static int complexity(AstNode node) {
    ComplexityVisitor visitor = node.is(PythonGrammar.FUNCDEF) ? new FunctionComplexityVisitor() : new ComplexityVisitor();
    visitor.scanNode(node);
    return visitor.complexity;
  }

  @Override
  public Set<AstNodeType> subscribedKinds() {
    Set<AstNodeType> set = new HashSet<>();
    set.add(PythonGrammar.FUNCDEF);
    set.add(PythonGrammar.FOR_STMT);
    set.add(PythonGrammar.WHILE_STMT);
    set.add(PythonKeyword.IF);
    set.add(PythonKeyword.AND);
    set.add(PythonKeyword.OR);
    return Collections.unmodifiableSet(set);
  }

  @Override
  public void visitFile(AstNode node) {
    complexity = 0;
  }

  @Override
  public void visitNode(AstNode node) {
    complexity++;
  }

  public int getComplexity() {
    return complexity;
  }

  private static class FunctionComplexityVisitor extends ComplexityVisitor {

    private int functionNestingLevel = 0;

    @Override
    public void visitNode(AstNode node) {
      if (node.is(PythonGrammar.FUNCDEF)) {
        functionNestingLevel++;
      }
      if (functionNestingLevel == 1) {
        super.visitNode(node);
      }
    }

    @Override
    public void leaveNode(AstNode node) {
      if (node.is(PythonGrammar.FUNCDEF)) {
        functionNestingLevel--;
      }
    }
  }

}
