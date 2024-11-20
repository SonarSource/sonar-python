/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;

class DocstringExtractorTest {

  private static final File BASE_DIR = new File("src/test/resources");

  private Map<Tree, StringLiteral> docstrings = new HashMap<>();

  @Test
  void test() {
    File file = new File(BASE_DIR, "docstring.py");
    TestPythonVisitorRunner.scanFile(file, new DocstringVisitor());

    assertDocstring(Tree.Kind.FILE_INPUT, 1, "\nThis is a module docstring\n");
    assertDocstring(Tree.Kind.FUNCDEF, 5, "This is a function docstring");
    assertDocstring(Tree.Kind.FUNCDEF, 12, " ");
    assertDocstring(Tree.Kind.FUNCDEF, 16, "This is a function docstring");
    assertDocstring(Tree.Kind.CLASSDEF, 20, "This is a class docstring");
    assertDocstring(Tree.Kind.FUNCDEF, 25, " This is a method docstring ");
    assertDocstring(Tree.Kind.FUNCDEF, 32, "This is stilla docstring");

    assertThat(docstrings).hasSize(12);
    assertThat(docstrings.values().stream().filter(Objects::nonNull)).hasSize(7);
  }

  private void assertDocstring(Tree.Kind kind, int line, String expectedDocstring) {
    StringLiteral docString = getDocstring(kind, line);
    assertThat(docString).as("docstring for AstNode of type " + kind + " at line " + line).isNotNull();
    assertThat(docString.trimmedQuotesValue()).isEqualTo(expectedDocstring);
  }

  private StringLiteral getDocstring(Tree.Kind kind, int line) {
    for (Map.Entry<Tree, StringLiteral> e : docstrings.entrySet()) {
      if (e.getKey().is(kind) && e.getKey().firstToken().line() == line) {
        return e.getValue();
      }
    }
    return null;
  }

  private class DocstringVisitor extends PythonVisitorCheck {

    @Override
    public void visitFileInput(FileInput fileInput) {
      docstrings.put(fileInput, DocstringExtractor.extractDocstring(fileInput.statements()));
      super.visitFileInput(fileInput);
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      docstrings.put(functionDef, DocstringExtractor.extractDocstring(functionDef.body()));
      super.visitFunctionDef(functionDef);
    }

    @Override
    public void visitClassDef(ClassDef classDef) {
      docstrings.put(classDef, DocstringExtractor.extractDocstring(classDef.body()));
      super.visitClassDef(classDef);
    }
  }
}
