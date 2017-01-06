/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.python;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import java.io.File;
import java.util.Map;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;

import static org.assertj.core.api.Assertions.assertThat;

public class DocstringVisitorTest {

  private static final File BASE_DIR = new File("src/test/resources");

  @SuppressWarnings("unchecked")
  @Test
  public void test() {
    File file = new File(BASE_DIR, "docstring.py");

    DocstringVisitor<Grammar> visitor = new DocstringVisitor<>();
    PythonAstScanner.scanSingleFile(file.getPath(), visitor);
    Map<AstNode, AstNode> docstrings = visitor.getDocstrings();

    String TRIPLE_QUOTES = "\"\"\"";

    assertDocstring(PythonGrammar.FILE_INPUT, 1, TRIPLE_QUOTES + "\nThis is a module docstring\n" + TRIPLE_QUOTES, docstrings);
    assertDocstring(PythonGrammar.FUNCDEF, 5, TRIPLE_QUOTES + "This is a function docstring" + TRIPLE_QUOTES, docstrings);
    assertDocstring(PythonGrammar.FUNCDEF, 13, TRIPLE_QUOTES + " " + TRIPLE_QUOTES, docstrings);
    assertDocstring(PythonGrammar.FUNCDEF, 18, "\"This is a function docstring\"", docstrings);
    assertDocstring(PythonGrammar.CLASSDEF, 22, TRIPLE_QUOTES + "This is a class docstring" + TRIPLE_QUOTES, docstrings);
    assertDocstring(PythonGrammar.FUNCDEF, 28, "''' This is a method docstring '''", docstrings);

    assertThat(docstrings).hasSize(10);
    assertThat(docstrings.values().stream().filter(ds -> ds != null)).hasSize(6);
  }

  private void assertDocstring(PythonGrammar nodeType, int line, String expectedDocString, Map<AstNode, AstNode> docstrings) {
    AstNode docString = getDocstring(nodeType, line, docstrings);
    assertThat(docString).as("docstring for AstNode of type " + nodeType + " at line " + line).isNotNull();
    assertThat(docString.getTokenValue()).isEqualTo(expectedDocString);
  }

  private AstNode getDocstring(PythonGrammar nodeType, int line, Map<AstNode, AstNode> docstrings) {
    for (AstNode e : docstrings.keySet()) {
      if (e.getType().equals(nodeType) && e.getTokenLine() == line) {
        return docstrings.get(e);
      }
    }
    return null;
  }

}
