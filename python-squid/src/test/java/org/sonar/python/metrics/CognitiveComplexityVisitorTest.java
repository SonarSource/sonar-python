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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.Trivia;
import java.io.File;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.junit.Test;
import org.sonar.python.PythonVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.api.PythonGrammar;

import static org.fest.assertions.Assertions.assertThat;

public class CognitiveComplexityVisitorTest {

  @Test
  public void file() {
    Map<Integer, String> complexityByLine = new TreeMap<>();
    CognitiveComplexityVisitor fileComplexityVisitor = new CognitiveComplexityVisitor(
      (node, message) -> complexityByLine.merge(node.getTokenLine(), message, (a, b) -> a + " " + b));

    StringBuilder comments = new StringBuilder();
    PythonVisitor functionAndCommentVisitor = new PythonVisitor() {
      @Override
      public Set<AstNodeType> subscribedKinds() {
        return new HashSet<>(Arrays.asList(PythonGrammar.FUNCDEF));
      }

      @Override
      public void visitNode(AstNode node) {
        if (!node.hasAncestor(PythonGrammar.FUNCDEF)) {
          int functionComplexity = CognitiveComplexityVisitor.complexity(node, null);
          complexityByLine.merge(node.getTokenLine(), "=" + functionComplexity, (a, b) -> a + " " + b);
        }
      }

      @Override
      public void visitToken(Token token) {
        for (Trivia trivia : token.getTrivia()) {
          if (trivia.isComment()) {
            String content = trivia.getToken().getValue().substring(1).trim();
            if (content.startsWith("=") || content.startsWith("+")) {
              comments.append("line " + trivia.getToken().getLine() + " " + content + "\n");
            }
          }
        }
      }
    };
    TestPythonVisitorRunner.scanFile(new File("src/test/resources/metrics/cognitive-complexities.py"), fileComplexityVisitor, functionAndCommentVisitor);
    assertThat(fileComplexityVisitor.getComplexity()).isEqualTo(89);

    StringBuilder complexityReport = new StringBuilder();
    complexityByLine.forEach((line, message) -> complexityReport.append("line " + line + " " + message + "\n"));
    assertThat(complexityReport.toString()).isEqualTo(comments.toString());
  }

}
