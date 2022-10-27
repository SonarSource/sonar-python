/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.quickfix;

import java.util.List;
import org.junit.Test;
import org.mockito.Mockito;
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

public class PythonTextEditTest {

  @Test
  public void insertBefore() {
    String textToInsert = "This is a replacement text";
    Token token = mockToken("token", 1, 7);

    PythonTextEdit textEdit = PythonTextEdit.insertBefore(token, textToInsert);
    assertThat(textEdit.replacementText()).isEqualTo(textToInsert);
    assertTextEditLocation(textEdit, 1, 7, 1, 7);
  }

  @Test
  public void insertAfter() {
    String textToInsert = "This is a replacement text";
    Token token = mockToken("token", 1, 7);

    PythonTextEdit textEdit = PythonTextEdit.insertAfter(token, textToInsert);
    assertThat(textEdit.replacementText()).isEqualTo(textToInsert);
    assertTextEditLocation(textEdit, 1, 12, 1, 12);
  }

  @Test
  public void replace() {
    String replacementText = "This is a replacement text";
    Token token = mockToken("token", 1, 7);

    PythonTextEdit textEdit = PythonTextEdit.replace(token, replacementText);
    assertThat(textEdit.replacementText()).isEqualTo(replacementText);
    assertTextEditLocation(textEdit, 1, 7, 1, 12);
  }

  @Test
  public void remove() {
    Token token = mockToken("token", 1, 7);

    PythonTextEdit textEdit = PythonTextEdit.remove(token);
    assertThat(textEdit.replacementText()).isEmpty();
    assertTextEditLocation(textEdit, 1, 7, 1, 12);
  }

  @Test
  public void replaceRange() {
    // Parsing 'a = (b and c)'
    Token token1 = mockToken("(", 1, 4);
    Token token2 = mockToken(")", 1, 12);

    PythonTextEdit textEdit = PythonTextEdit.replaceRange(token1, token2, "b and c");
    assertThat(textEdit.replacementText()).isEqualTo("b and c");
    assertTextEditLocation(textEdit, 1, 4, 1, 13);
  }

  @Test
  public void insertLineBefore() {
    Token token = mockToken("tree", 1, 4);

    PythonTextEdit textEdit = PythonTextEdit.insertLineBefore(token, "firstLine\n    secondLineWithIndent\n");
    assertThat(textEdit.replacementText()).isEqualTo("firstLine\n        secondLineWithIndent\n    ");
    assertTextEditLocation(textEdit, 1, 4, 1, 4);
  }

  @Test
  public void shiftLeft() {
    FileInput file = parse(
      "def foo():",
      "    a = 1; b = 2",
      " # comment",
      "    c = 2"
    );
    StatementList functionBody = ((FunctionDef) PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.FUNCDEF))).body();

    List<PythonTextEdit> textEdits = PythonTextEdit.shiftLeft(functionBody);
    assertThat(textEdits).hasSize(2);
    textEdits.forEach(textEdit -> {
      assertThat(textEdit.startLineOffset()).isZero();
      assertThat(textEdit.replacementText()).isEmpty();
      assertThat(textEdit.endLineOffset()).isEqualTo(4);
    });
  }

  @Test
  public void removeUntil() {
    FileInput file = parse(
      "def foo():",
      "    a = 1",
      " # comment",
      "    b = 2"
    );

    StatementList functionBody = ((FunctionDef) PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.FUNCDEF))).body();
    Statement lastStatement = ListUtils.getLast(functionBody.statements());

    PythonTextEdit textEdit = PythonTextEdit.removeUntil(functionBody, lastStatement);
    assertThat(textEdit.replacementText()).isEmpty();
    assertTextEditLocation(textEdit, 2, 4, 4, 4);
  }

  @Test
  public void testRenameAllUsages() {
    FileInput file = parse(
      "def foo(bar):",
      "    print(bar)",
      " # comment",
      "    b = 2"
    );
    Parameter parameter = PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.PARAMETER));

    List<PythonTextEdit> textEdits = PythonTextEdit.renameAllUsages(parameter.name(), "xxx");

    assertThat(textEdits).containsExactly(
      new PythonTextEdit("xxx", 1, 8, 1, 11),
      new PythonTextEdit("xxx", 2, 10, 2, 13)
    );
  }

  @Test
  public void test_insertLineAfter_without_indent() {
    FileInput file = parse("foo()");
    CallExpression call = PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.CALL_EXPR));

    PythonTextEdit textEdit = PythonTextEdit.insertLineAfter(call, call, "bar");
    assertThat(textEdit).isEqualTo(new PythonTextEdit("\nbar", 1,3,1,3));
  }

  @Test
  public void test_insertLineAfter_with_indent() {
    FileInput file = parse(
      "def foo():",
      "    pass"
    );
    FunctionDef functionDef = PythonTestUtils.getFirstDescendant(file, descendant -> descendant.is(Tree.Kind.FUNCDEF));
    StatementList functionBody = functionDef.body();

    PythonTextEdit textEdit = PythonTextEdit.insertLineAfter(functionDef.colon(), functionBody, "bar");
    assertThat(textEdit).isEqualTo(new PythonTextEdit("\n    bar", 1,10,1,10));
  }

  @Test
  public void equals() {
    PythonTextEdit edit = new PythonTextEdit("", 0, 0, 1, 1);
    assertThat(edit.equals(edit)).isTrue();
    assertThat(edit.equals(null)).isFalse();
    assertThat(edit.equals(new Object())).isFalse();

    assertThat(edit.equals(new PythonTextEdit("", 0, 0, 1, 1))).isTrue();
    assertThat(edit.equals(new PythonTextEdit("",1, 0, 1, 1))).isFalse();
    assertThat(edit.equals(new PythonTextEdit("",0, 1, 1, 1))).isFalse();
    assertThat(edit.equals(new PythonTextEdit("",0, 0, 0, 1))).isFalse();
    assertThat(edit.equals(new PythonTextEdit("",0, 0, 1, 0))).isFalse();
    assertThat(edit.equals(new PythonTextEdit("a", 0, 0, 1, 1))).isFalse();
  }

  @Test
  public void test_hashCode() {
    PythonTextEdit edit = new PythonTextEdit("", 0, 0, 1, 1);
    assertThat(edit)
      .hasSameHashCodeAs(edit)
      .hasSameHashCodeAs(new PythonTextEdit("", 0, 0, 1, 1))
      .doesNotHaveSameHashCodeAs(new Object())
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("",1, 0, 1, 1))
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("",0, 1, 1, 1))
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("",0, 0, 0, 1))
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("",0, 0, 1, 0))
      .doesNotHaveSameHashCodeAs(new PythonTextEdit("a", 0, 0, 1, 1))
    ;
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
}
