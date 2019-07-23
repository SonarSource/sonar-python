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
package org.sonar.python.metrics;

import com.intellij.psi.PsiComment;
import com.intellij.psi.PsiElement;
import com.intellij.psi.util.PsiTreeUtil;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyFunction;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import java.io.File;
import java.util.Map;
import java.util.TreeMap;
import org.junit.Test;
import org.sonar.python.frontend.PythonParser;
import org.sonar.python.frontend.PythonTokenLocation;

import static org.fest.assertions.Assertions.assertThat;

public class CognitiveComplexityVisitorTest {

  @Test
  public void file() {
    Map<Integer, String> complexityByLine = new TreeMap<>();
    CognitiveComplexityVisitor fileComplexityVisitor = new CognitiveComplexityVisitor(
      (node, message) -> complexityByLine.merge(line(node), message, (a, b) -> a + " " + b));

    StringBuilder comments = new StringBuilder();
    PyRecursiveElementVisitor functionAndCommentVisitor = new PyRecursiveElementVisitor() {

      @Override
      public void visitPyFunction(PyFunction node) {
        if (PsiTreeUtil.getParentOfType(node, PyFunction.class) == null) {
          int functionComplexity = CognitiveComplexityVisitor.complexity(node, null);
          complexityByLine.merge(line(node), "=" + functionComplexity, (a, b) -> a + " " + b);
        }
        super.visitPyFunction(node);
      }

      @Override
      public void visitComment(PsiComment comment) {
        String content = comment.getText().substring(1).trim();
        if (content.startsWith("=") || content.startsWith("+")) {
          comments.append("line " + line(comment) + " " + content + "\n");
        }
      }
    };
    PyFile pyFile = PythonParser.parse(new File("src/test/resources/metrics/cognitive-complexities.py"));
    pyFile.accept(fileComplexityVisitor);
    pyFile.accept(functionAndCommentVisitor);
    assertThat(fileComplexityVisitor.getComplexity()).isEqualTo(91);

    StringBuilder complexityReport = new StringBuilder();
    complexityByLine.forEach((line, message) -> complexityReport.append("line " + line + " " + message + "\n"));
    assertThat(complexityReport.toString()).isEqualTo(comments.toString());
  }

  private static int line(PsiElement element) {
    return new PythonTokenLocation(element).startLine();
  }

}
