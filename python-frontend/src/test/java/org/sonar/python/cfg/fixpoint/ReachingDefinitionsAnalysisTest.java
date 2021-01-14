/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.cfg.fixpoint;

import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXPRESSION_STMT;
import static org.sonar.python.PythonTestUtils.getFirstDescendant;
import static org.sonar.python.PythonTestUtils.getLastDescendant;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.lastExpressionInFunction;
import static org.sonar.python.PythonTestUtils.parse;

public class ReachingDefinitionsAnalysisTest {
  private final PythonFile file = Mockito.mock(PythonFile.class, "file1.py");
  ReachingDefinitionsAnalysis analysis = new ReachingDefinitionsAnalysis(file);

  @Test
  public void valuesAtLocation_single_assignment() {
    Name x = (Name) lastExpressionInFunction("x = 42; x");
    assertThat(analysis.valuesAtLocation(x)).extracting(ReachingDefinitionsAnalysisTest::getValueAsString).containsExactly("42");
  }

  @Test
  public void valuesAtLocation_multiple_assignments() {
    Name x = (Name) lastExpressionInFunction("x = 1; x = 2; x");
    assertThat(analysis.valuesAtLocation(x)).extracting(ReachingDefinitionsAnalysisTest::getValueAsString).containsExactly("2");
  }

  @Test
  public void valuesAtLocation_branches() {
    Name x = (Name) lastExpressionInFunction(
      "if p:",
      "  x = 1",
      "else:",
      "  x = 2",
      "x"
    );
    assertThat(analysis.valuesAtLocation(x)).extracting(ReachingDefinitionsAnalysisTest::getValueAsString).containsExactlyInAnyOrder("1", "2");
  }

  @Test
  public void valuesAtLocation_outside_function() {
    Name x = (Name) lastExpression("x = 42; x");
    assertThat(analysis.valuesAtLocation(x)).isEmpty();
  }

  @Test
  public void valuesAtLocation_invalid_cfg() {
    Name x = (Name) lastExpressionInFunction("x = 42", "break", "x");
    assertThat(analysis.valuesAtLocation(x)).isEmpty();
  }

  @Test
  public void valuesAtLocation_no_name_assignment() {
    Name x = (Name) lastExpressionInFunction("x.foo = 42", "x");
    assertThat(analysis.valuesAtLocation(x)).isEmpty();
  }

  @Test
  public void valuesAtLocation_assignment_lhs() {
    Name x = (Name) lastExpressionInFunction("x = y = 42", "x");
    assertThat(analysis.valuesAtLocation(x)).isEmpty();
  }

  @Test
  public void loop_with_conditions() {
    FileInput fileInput = parse(
      "def f():",
      "  for i in range(3):",
      "    if i > 1:",
      "      x",
      "    elif i==1:",
      "      x = 2",
      "    else:",
      "      x = 1",
      "      x"
      );
    Name x = ((Name) ((ExpressionStatement) getFirstDescendant(fileInput, tree -> tree.is(EXPRESSION_STMT))).expressions().get(0));
    assertThat(analysis.valuesAtLocation(x)).extracting(ReachingDefinitionsAnalysisTest::getValueAsString).containsExactlyInAnyOrder("1", "2");

    x = ((Name) ((ExpressionStatement) getLastDescendant(fileInput, tree -> tree.is(EXPRESSION_STMT))).expressions().get(0));
    assertThat(analysis.valuesAtLocation(x)).extracting(ReachingDefinitionsAnalysisTest::getValueAsString).containsExactlyInAnyOrder("1");

    fileInput = parse(
      "def f():",
      "  for i in range(3):",
      "    x = 1",
      "    if i > 1:",
      "      x = 2",
      "    x"
    );
    x = ((Name) ((ExpressionStatement) getLastDescendant(fileInput, tree -> tree.is(EXPRESSION_STMT))).expressions().get(0));
    assertThat(analysis.valuesAtLocation(x)).extracting(ReachingDefinitionsAnalysisTest::getValueAsString).containsExactlyInAnyOrder("1", "2");
  }

  @Test
  public void ignore_outer_scope() {
    FileInput fileInput = parse(
      "def f():",
      "  x = 42",
      "  def inner():",
      "    x"
    );
    Name x = ((Name) ((ExpressionStatement) getLastDescendant(fileInput, tree -> tree.is(EXPRESSION_STMT))).expressions().get(0));
    assertThat(analysis.valuesAtLocation(x)).isEmpty();
  }

  @Test
  public void compound_assignments() {
    Name x = (Name) lastExpressionInFunction("x = 42; x += 1; x");
    assertThat(analysis.valuesAtLocation(x)).isEmpty();
  }

  @Test
  public void try_stmt() {
    Name x = (Name) lastExpressionInFunction(
      "x = 1",
      "try:",
      "  x = 2",
      "except:",
      "  pass",
      "x"
    );
    assertThat(analysis.valuesAtLocation(x)).isEmpty();
  }
  
  private static String getValueAsString(Expression expression) {
    return ((NumericLiteral) expression).valueAsString();
  }
}
