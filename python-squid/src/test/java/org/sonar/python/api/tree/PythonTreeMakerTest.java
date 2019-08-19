/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.api.tree;

import com.sonar.sslr.api.AstNode;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonTreeMakerTest extends RuleTest {

  @Test
  public void fileInputTreeOnEmptyFile() {
    AstNode astNode = p.parse("");
    PyFileInputTree pyTree = new PythonTreeMaker().fileInput(astNode);
    assertThat(pyTree.statements()).isEmpty();
  }

  @Test
  public void IfStatement() {
    setRootRule(PythonGrammar.IF_STMT);
    AstNode astNode = p.parse("if x: pass");
    PyIfStatementTree pyIfStatementTree = new PythonTreeMaker().ifStatement(astNode);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);

    astNode = p.parse("if x: pass\nelse: pass");
    pyIfStatementTree = new PythonTreeMaker().ifStatement(astNode);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.elseBranch()).isNotNull();

    astNode = p.parse("if x: pass\nelif y: pass");
    pyIfStatementTree = new PythonTreeMaker().ifStatement(astNode);
    assertThat(pyIfStatementTree.keyword().getValue()).isEqualTo("if");
    assertThat(pyIfStatementTree.condition()).isInstanceOf(PyExpressionTree.class);
    assertThat(pyIfStatementTree.elifBranches()).isNotEmpty();
    PyIfStatementTree elif = pyIfStatementTree.elifBranches().get(0);
    assertThat(elif.keyword().getValue()).isEqualTo("elif");
  }

}
