/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

/**
 * For internal use only. Can not be used outside SonarPython analyzer.
 */
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

  /**
   * Insert a line with the same offset as the given tree, before the given tree.
   * Offset is applied to multiline insertions.
   */
  public static PythonTextEdit insertLineBefore(Tree tree, String textToInsert) {
    String lineOffset = offset(tree);
    textToInsert += "\n";
    String textWithOffset = textToInsert.replace("\n", "\n" + lineOffset);
    return insertBefore(tree, textWithOffset);
  }

  /**
   * Insert a line with the same offset as the given tree, after the given tree.
   * Offset is applied to multiline insertions and calculated by the reference tree.
   */
  public static PythonTextEdit insertLineAfter(Tree tree, Tree indentReference, String textToInsert) {
    String lineOffset = offset(indentReference);
    String textWithOffset = "\n" + lineOffset + textToInsert.replace("\n", "\n" + lineOffset);
    return insertAfter(tree, textWithOffset);
  }

  private static String offset(Tree referenceTree) {
    return " ".repeat(referenceTree.firstToken().column());
  }

  public static PythonTextEdit insertBefore(Tree tree, String textToInsert) {
    Token token = tree.firstToken();
    return insertAtPosition(token.line(), token.column(), textToInsert);
  }

  public static PythonTextEdit insertAfter(Tree tree, String textToInsert) {
    Token token = tree.firstToken();
    int lengthToken = token.value().length();
    return insertAtPosition(token.line(), token.column() + lengthToken, textToInsert);
  }

  public static PythonTextEdit insertAtPosition(int line, int column, String textToInsert) {
    return new PythonTextEdit(textToInsert, line, column, line, column);
  }

  public static PythonTextEdit replace(Tree toReplace, String replacementText) {
    return replaceRange(toReplace, toReplace, replacementText);
  }

  public static PythonTextEdit replaceRange(Tree start, Tree end, String replacementText) {
    Token first = start.firstToken();
    Token last = end.lastToken();
    return new PythonTextEdit(replacementText, first.line(), first.column(), last.line(), last.column() + last.value().length());
  }

  /**
   * Shift body statements to be on same level as the parent statement
   * Filter out text edits which apply on the same line which could show up with multiple statements on the same line
   */
  public static List<PythonTextEdit> shiftLeft(StatementList statementList) {
    int offset = statementList.firstToken().column() - statementList.parent().firstToken().column();
    return statementList.statements().stream()
      .map(statement -> shiftLeft(statement, offset))
      .flatMap(List::stream)
      .distinct()
      .collect(Collectors.toList());
  }

  /**
   * Shift single statement of a statement list by the given offset.
   * Take care about child statements by collecting all child tokens and shift each line once.
   */
  private static List<PythonTextEdit> shiftLeft(Tree tree, int offset) {
    return TreeUtils.tokens(tree).stream()
      .filter(token -> token.column() >= offset)
      .map(Token::line)
      .distinct()
      .map(line -> removeRange(line, 0, line, offset))
      .collect(Collectors.toList());
  }

  public static PythonTextEdit removeRange(int startLine, int startColumn, int endLine, int endColumn) {
    return new PythonTextEdit("", startLine, startColumn, endLine, endColumn);
  }

  /**
   * Remove range including the start token until the beginning of the end tree's first token.
   * This is useful to remove and shift multiple statement over multiple lines.
   */
  public static PythonTextEdit removeUntil(Tree start, Tree until) {
    return removeRange(start.firstToken().line(), start.firstToken().column(), until.firstToken().line(), until.firstToken().column());
  }

  public static PythonTextEdit removeStatement(Statement statement) {
    Token firstTokenOfStmt = statement.firstToken();
    Token lastTokenOfStmt = TreeUtils.getTreeSeparatorOrLastToken(statement);

    List<Tree> siblings = statement.parent().children();
    int indexOfTree = siblings.indexOf(statement);
    Tree previous = indexOfTree > 0 ? siblings.get(indexOfTree - 1) : null;
    Tree next = indexOfTree < siblings.size() - 1 ? siblings.get(indexOfTree + 1) : null;

    // Statement is the single element in the block
    // Replace by `pass` keyword
    if (previous == null && next == null) {
      return PythonTextEdit.replace(statement, "pass");
    }

    boolean hasPreviousSiblingOnLine = previous != null && firstTokenOfStmt.line() == TreeUtils.getTreeSeparatorOrLastToken(previous.lastToken()).line();
    boolean hasNextSiblingOnLine = next != null && lastTokenOfStmt.line() == next.firstToken().line();

    if (hasNextSiblingOnLine) {
      // Statement is first on the line or between at least two statements
      // Remove from first token to last toke of statement
      Token firstNextToken = next.firstToken();
      return PythonTextEdit.removeRange(firstTokenOfStmt.line(), firstTokenOfStmt.column(), firstNextToken.line(), firstNextToken.column());
    } else if (hasPreviousSiblingOnLine) {
      // Statement is last on the line and has one or more previous statement on the line
      // Remove from last token or separator of previous statement to avoid trailing white spaces
      // Keep the line break to ensure elements on the next line don't get pushed to the current line
      Token lastPreviousToken = TreeUtils.getTreeSeparatorOrLastToken(previous);
      return PythonTextEdit.removeRange(lastPreviousToken.line(), getEndColumn(lastPreviousToken), lastPreviousToken.line(), getEndColumn(lastTokenOfStmt) - 1);
    } else {
      // Statement is single on the line
      // Remove the entire line including indent and line break
      return PythonTextEdit.removeRange(firstTokenOfStmt.line(), 0, lastTokenOfStmt.line(), getEndColumn(lastTokenOfStmt));
    }
  }

  /**
   * Token can be longer than a single character so we have to add the value length to determine the end column of the token
   */
  private static int getEndColumn(Token token) {
    return token.column() + token.value().length();
  }

  public static PythonTextEdit remove(Tree toRemove) {
    return replace(toRemove, "");
  }

  /**
   * Returns a list of all replacements for the symbol to new name.
   * If there is no usages empty list is returned.
   */
  public static List<PythonTextEdit> renameAllUsages(HasSymbol node, String newName) {
    Symbol symbol = node.symbol();
    List<Usage> usages = symbol != null ? symbol.usages() : Collections.emptyList();
    List<PythonTextEdit> result = new LinkedList<>();
    for(Usage usage: usages) {
      PythonTextEdit text = PythonTextEdit.replace(usage.tree().firstToken(), newName);
      result.add(text);
    }
    return result;
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

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    PythonTextEdit that = (PythonTextEdit) o;
    return startLine == that.startLine && startLineOffset == that.startLineOffset && endLine == that.endLine
      && endLineOffset == that.endLineOffset && Objects.equals(message, that.message);
  }

  @Override
  public int hashCode() {
    return Objects.hash(message, startLine, startLineOffset, endLine, endLineOffset);
  }

  @Override
  public String toString() {
    return "PythonTextEdit{" +
      "message='" + message + '\'' +
      ", startLine=" + startLine +
      ", startLineOffset=" + startLineOffset +
      ", endLine=" + endLine +
      ", endLineOffset=" + endLineOffset +
      '}';
  }
}
