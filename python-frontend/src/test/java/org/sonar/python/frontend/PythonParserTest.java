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
package org.sonar.python.frontend;

import com.intellij.openapi.editor.Document;
import com.intellij.psi.PsiElement;
import com.intellij.psi.util.PsiTreeUtil;
import com.jetbrains.python.psi.PyCallExpression;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyFormattedStringElement;
import com.jetbrains.python.psi.PyParenthesizedExpression;
import com.jetbrains.python.psi.PyPrintStatement;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import com.jetbrains.python.psi.PyStatement;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonParserTest {

  private final PythonParser parser = new PythonParser();

  @Test
  public void print_statement() {
    PyFile pyFile = parser.parse("print 42");
    PyStatement statement = pyFile.getStatements().get(0);
    assertThat(statement).isInstanceOf(PyPrintStatement.class);
  }

  @Test
  public void print_with_parentheses() {
    PyFile pyFile = parser.parse("print(42)");
    PyStatement statement = pyFile.getStatements().get(0);
    assertThat(statement).isInstanceOf(PyPrintStatement.class);
    assertThat(statement.getLastChild()).isInstanceOf(PyParenthesizedExpression.class);
  }

  @Test
  public void call_expressions() {
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

  @Test
  public void python3_f_string() {
    PyFile pyFile = parser.parse("f\"Hello {name}!\"");
    PyFormattedStringElement stringElement = PsiTreeUtil.getParentOfType(pyFile.findElementAt(0), PyFormattedStringElement.class);
    assertThat(stringElement.getDecodedFragments().stream().map(pair -> pair.second)).containsExactly("Hello ", "{name}", "!");
  }

  @Test
  public void line_separator() {
    PyFile pyFile = parser.parse("print(42)\r\nprint(43)");
    PsiElement secondPrint = pyFile.getLastChild();
    Document document = secondPrint.getContainingFile().getViewProvider().getDocument();
    assertThat(document.getLineNumber(secondPrint.getTextRange().getStartOffset())).isEqualTo(1);

    pyFile = parser.parse("print(42)\rprint(43)");
    secondPrint = pyFile.getLastChild();
    document = secondPrint.getContainingFile().getViewProvider().getDocument();
    assertThat(document.getLineNumber(secondPrint.getTextRange().getStartOffset())).isEqualTo(1);

    pyFile = parser.parse("print(42)\nprint(43)");
    secondPrint = pyFile.getLastChild();
    document = secondPrint.getContainingFile().getViewProvider().getDocument();
    assertThat(document.getLineNumber(secondPrint.getTextRange().getStartOffset())).isEqualTo(1);
  }
}
