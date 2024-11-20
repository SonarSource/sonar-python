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
package org.sonar.python.quickfix;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonarsource.analyzer.commons.collections.ListUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.when;
import static org.sonar.python.PythonTestUtils.parse;

class TextEditUtilsTest {

  @Test
  void insertBefore() {
    String textToInsert = "This is a replacement text";
    Token token = mockToken("token", 1, 7);

    PythonTextEdit textEdit = TextEditUtils.insertBefore(token, textToInsert);
    assertThat(textEdit.replacementText()).isEqualTo(textToInsert);
    assertTextEditLocation(textEdit, 1, 7, 1, 7);
  }

  @Test
  void insertAfter() {
    String textToInsert = "This is a replacement text";
    Token token = mockToken("token", 1, 7);

    PythonTextEdit textEdit = TextEditUtils.insertAfter(token, textToInsert);
    assertThat(textEdit.replacementText()).isEqualTo(textToInsert);
    assertTextEditLocation(textEdit, 1, 12, 1, 12);
  }

  @Test
  void replace() {
    String replacementText = "This is a replacement text";
    Token token = mockToken("token", 1, 7);

    PythonTextEdit textEdit = TextEditUtils.replace(token, replacementText);
    assertThat(textEdit.replacementText()).isEqualTo(replacementText);
    assertTextEditLocation(textEdit, 1, 7, 1, 12);
  }

  @Test
  void remove() {
    Token token = mockToken("token", 1, 7);

    PythonTextEdit textEdit = TextEditUtils.remove(token);
    assertThat(textEdit.replacementText()).isEmpty();
    assertTextEditLocation(textEdit, 1, 7, 1, 12);
  }

  @Test
  void replaceRange() {
    // Parsing 'a = (b and c)'
    Token token1 = mockToken("(", 1, 4);
    Token token2 = mockToken(")", 1, 12);

    PythonTextEdit textEdit = TextEditUtils.replaceRange(token1, token2, "b and c");
    assertThat(textEdit.replacementText()).isEqualTo("b and c");
    assertTextEditLocation(textEdit, 1, 4, 1, 13);
  }

  @Test
  void insertLineBefore() {
    Token token = mockToken("tree", 1, 4);

    PythonTextEdit textEdit = TextEditUtils.insertLineBefore(token, "firstLine\n    secondLineWithIndent");
    assertThat(textEdit.replacementText()).isEqualTo("firstLine\n        secondLineWithIndent\n    ");
    assertTextEditLocation(textEdit, 1, 4, 1, 4);
  }

  @Test
  void shiftLeft() {
    FileInput file = parse(
      "def foo():",
      "    a = 1; b = 2",
      " # comment",
      "    c = 2"
    );
    StatementList functionBody = getFunctionBody(file);

    List<PythonTextEdit> textEdits = TextEditUtils.shiftLeft(functionBody);
    assertThat(textEdits).hasSize(2);
    textEdits.forEach(textEdit -> {
      assertThat(textEdit.startLineOffset()).isZero();
      assertThat(textEdit.replacementText()).isEmpty();
      assertThat(textEdit.endLineOffset()).isEqualTo(4);
    });
  }

  @Test
  void removeUntil() {
    FileInput file = parse(
      "def foo():",
      "    a = 1",
      " # comment",
      "    b = 2"
    );

    StatementList functionBody = getFunctionBody(file);
    Statement lastStatement = ListUtils.getLast(functionBody.statements());

    PythonTextEdit textEdit = TextEditUtils.removeUntil(functionBody, lastStatement);
    assertThat(textEdit.replacementText()).isEmpty();
    assertTextEditLocation(textEdit, 2, 4, 4, 4);
  }

  @Test
  void testRenameAllUsages() {
    FileInput file = parse(
      "def foo(bar):",
      "    print(bar)",
      " # comment",
      "    b = 2"
    );
    Parameter parameter = PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.PARAMETER));

    List<PythonTextEdit> textEdits = TextEditUtils.renameAllUsages(parameter.name(), "xxx");

    assertThat(textEdits).containsExactly(
      new PythonTextEdit("xxx", 1, 8, 1, 11),
      new PythonTextEdit("xxx", 2, 10, 2, 13)
    );
  }

  @Test
  void test_insertLineAfter_without_indent() {
    FileInput file = parse("foo()");
    CallExpression call = PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.CALL_EXPR));

    PythonTextEdit textEdit = TextEditUtils.insertLineAfter(call, call, "bar");
    assertThat(textEdit).isEqualTo(new PythonTextEdit("\nbar", 1,3,1,3));
  }

  @Test
  void test_insertLineAfter_with_indent() {
    FileInput file = parse(
      "def foo():",
      "    pass"
    );
    FunctionDef functionDef = PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.FUNCDEF));
    StatementList functionBody = functionDef.body();

    PythonTextEdit textEdit = TextEditUtils.insertLineAfter(functionDef.colon(), functionBody, "bar");
    assertThat(textEdit).isEqualTo(new PythonTextEdit("\n    bar", 1,10,1,10));
  }

  @Test
  void removeStatement_with_single_statement_will_replace_with_pass() {
    FileInput file = parse("def foo():",
      "    a = 1");
    StatementList functionBody = getFunctionBody(file);

    PythonTextEdit textEdit = TextEditUtils.removeStatement(functionBody.statements().get(0));
    assertThat(textEdit).isEqualTo(new PythonTextEdit("pass", 2, 4, 2, 9));
  }

  @Test
  void removeStatement_with_two_statement_will_remove_indent_and_line_breaks() {
    FileInput file = parse("def foo():",
      "    a = 1",
      "    b = 2");
    StatementList functionBody = getFunctionBody(file);

    PythonTextEdit firstTextEdit = TextEditUtils.removeStatement(functionBody.statements().get(0));
    assertThat(firstTextEdit).isEqualTo(new PythonTextEdit("", 2, 0, 2, 10));

    PythonTextEdit secondTextEdit = TextEditUtils.removeStatement(functionBody.statements().get(1));
    assertThat(secondTextEdit).isEqualTo(new PythonTextEdit("", 3, 0, 3, 9));
  }

  @Test
  void removeStatement_with_statement_with_separator_will_remove_all() {
    FileInput file = parse("def foo():",
      "    a = 1;",
      "    b = 2");
    StatementList functionBody = getFunctionBody(file);

    PythonTextEdit firstTextEdit = TextEditUtils.removeStatement(functionBody.statements().get(0));
    assertThat(firstTextEdit).isEqualTo(new PythonTextEdit("", 2, 0, 2, 11));
  }

  @Test
  void removeStatement_with_tree_statement_on_same_line_will_not_remove_indent_nor_line_breaks() {
    FileInput file = parse("def foo():",
      "    a = 1; b = 2; c = 3;");
    StatementList functionBody = getFunctionBody(file);

    PythonTextEdit firstTextEdit = TextEditUtils.removeStatement(functionBody.statements().get(0));
    assertThat(firstTextEdit).isEqualTo(new PythonTextEdit("", 2, 4, 2, 11));

    PythonTextEdit secondTextEdit = TextEditUtils.removeStatement(functionBody.statements().get(1));
    assertThat(secondTextEdit).isEqualTo(new PythonTextEdit("", 2, 11, 2, 18));

    PythonTextEdit thirdTextEdit = TextEditUtils.removeStatement(functionBody.statements().get(2));
    assertThat(thirdTextEdit).isEqualTo(new PythonTextEdit("", 2, 17, 2, 23));
  }

  private void assertTextEditLocation(PythonTextEdit textEdit, int startLine, int startLineOffset, int endLine, int endLineOffset) {
    assertThat(textEdit.startLine()).isEqualTo(startLine);
    assertThat(textEdit.startLineOffset()).isEqualTo(startLineOffset);
    assertThat(textEdit.endLine()).isEqualTo(endLine);
    assertThat(textEdit.endLineOffset()).isEqualTo(endLineOffset);
  }

  private static Token mockToken(String value, int line, int column) {
    Token token = Mockito.mock(Token.class);
    when(token.firstToken()).thenReturn(token);
    when(token.lastToken()).thenReturn(token);

    when(token.value()).thenReturn(value);
    when(token.line()).thenReturn(line);
    when(token.column()).thenReturn(column);

    return token;
  }

  private static StatementList getFunctionBody(FileInput file) {
    return ((FunctionDef) PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.FUNCDEF))).body();
  }
}
