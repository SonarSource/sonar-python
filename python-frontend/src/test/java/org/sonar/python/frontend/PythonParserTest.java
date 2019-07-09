package org.sonar.python.frontend;/*
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
import com.jetbrains.python.psi.PyCallExpression;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonParserTest {

  @Test
  public void parse() {
    PythonParser parser = new PythonParser();
    PyFile pyFile = parser.parse("s = 'abc'\nfoo(s, 42)\nbar(43)");
    List<String> callExpressions = new ArrayList<>();
    pyFile.accept(new PyRecursiveElementVisitor() {
      @Override
      public void visitPyCallExpression(PyCallExpression node) {
        callExpressions.add(node + "::" + node.getArguments(null));
        super.visitPyCallExpression(node);
      }
    });
    assertThat(callExpressions).containsExactly(
      "PyCallExpression: foo::[PyReferenceExpression: s, PyNumericLiteralExpression]",
      "PyCallExpression: bar::[PyNumericLiteralExpression]");
  }
}
