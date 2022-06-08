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
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;

public class PythonTextEdit {

  private final String message;
  private final int startLine;
  private final int startLineOffset;
  private final int endLine;
  private final int endLineOffset;

  public PythonTextEdit(String message, int startLine, int startLineOffset, int endLine, int endLineOffset) {
    this.message = message;
    this.startLine = startLine;
    this.startLineOffset = startLineOffset;
    this.endLine = endLine;
    this.endLineOffset = endLineOffset;
  }

  public static PythonTextEdit insertBefore(Tree tree, String textToInsert) {
    Token token = tree.firstToken();
    return new PythonTextEdit(textToInsert, token.line(), token.column(), token.line(), token.column());
  }

  public static PythonTextEdit insertAfter(Tree tree, String textToInsert) {
    Token token = tree.firstToken();
    int lengthToken = token.value().length();
    return new PythonTextEdit(textToInsert, token.line(), token.column() + lengthToken, token.line(), token.column() + lengthToken);
  }

  public static PythonTextEdit replace(Tree toReplace, String replacementText) {
    Token token = toReplace.firstToken();
    Token last = token.lastToken();
    return new PythonTextEdit(replacementText, token.line(), token.column(), last.line(), last.column() + last.value().length());
  }

  public static PythonTextEdit replaceRange(Tree start, Tree end, String replacementText) {
    Token first = start.firstToken();
    Token last = end.lastToken();
    return new PythonTextEdit(replacementText, first.line(), first.column(), last.line(), last.column() + last.value().length());
  }

  public static PythonTextEdit remove(Tree toRemove) {
    return replace(toRemove, "");
  }

  public static PythonTextEdit removeDeadStore(Tree tree) {
    List<Tree> childrenOfParent = tree.parent().children();
    if (childrenOfParent.size() == 1) {
      return remove(tree);
    }

    Token first = tree.firstToken();
    int i = childrenOfParent.indexOf(tree);
    if (i == childrenOfParent.size() - 1) {
      Token previous = childrenOfParent.get(i - 1).lastToken();
      first = tree.lastToken();
      // Replace from the end of the previous token (will also remove the separator and trailing whitespaces) until the end of the current token
      return new PythonTextEdit("", previous.line(), previous.column() + previous.value().length(), first.line(), first.column() + first.value().length());
    }
    Token next = childrenOfParent.get(i + 1).firstToken();
    // Remove from the start of the current tokenuntil the next token
    return new PythonTextEdit("", first.line(), first.column(), next.line(), next.column());
  }

  public String replacementText() {
    return message;
  }

  public int startLine() {
    return startLine;
  }

  public int startLineOffset() {
    return startLineOffset;
  }

  public int endLine() {
    return endLine;
  }

  public int endLineOffset() {
    return endLineOffset;
  }
}
