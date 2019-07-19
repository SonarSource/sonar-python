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
import com.intellij.psi.PsiWhiteSpace;
import com.intellij.psi.impl.source.tree.LeafPsiElement;
import com.jetbrains.python.PyTokenTypes;
import com.jetbrains.python.psi.PyExceptPart;
import com.jetbrains.python.psi.PyExpression;
import com.jetbrains.python.psi.PyExpressionStatement;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyIfPart;
import com.jetbrains.python.psi.PyIfStatement;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import com.jetbrains.python.psi.PyStatement;
import com.jetbrains.python.psi.PyStringLiteralExpression;
import com.jetbrains.python.psi.PyTryExceptStatement;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.sonar.python.frontend.PythonTokenLocation;

public class MetricsVisitor extends PyRecursiveElementVisitor {

  private final boolean ignoreHeaderComments;
  private Set<Integer> linesOfCode = new HashSet<>();
  private Set<Integer> executableLines = new HashSet<>();
  private Set<Integer> linesOfComments = new HashSet<>();
  private Set<Integer> noSonar = new HashSet<>();
  private boolean firstNonCommentSeen = false;

  public MetricsVisitor(boolean ignoreHeaderComments) {
    this.ignoreHeaderComments = ignoreHeaderComments;
  }

  @Override
  public void visitElement(PsiElement element) {
    // track the first meaningful element which is not a comment to handle ignoreHeaderComments
    if (!((element instanceof PyFile) || (element instanceof PsiComment) || (element instanceof PsiWhiteSpace))) {
      firstNonCommentSeen = true;
    }

    // track lines of code
    if (element instanceof LeafPsiElement
      && !(element instanceof PsiWhiteSpace)
      && !(element instanceof PsiComment)
      && element.getNode().getElementType() != PyTokenTypes.DOCSTRING) {
      PythonTokenLocation location = new PythonTokenLocation(element);
      for (int line = location.startLine(); line <= location.endLine(); line++) {
        linesOfCode.add(line);
      }
    }

    // track executable lines of code
    if (element instanceof PyStatement) {
      handlePyStatement((PyStatement) element);
    }

    // track lines of comments and no_sonar lines
    if (element instanceof PsiComment && (firstNonCommentSeen || !ignoreHeaderComments)) {
      handleComment(((PsiComment) element));
    }

    super.visitElement(element);
  }

  private void handlePyStatement(PyStatement element) {
    if (!isDocString(element)) {
      executableLines.add(new PythonTokenLocation(element).startLine());
    }
    if (element instanceof PyIfStatement) {
      for (PyIfPart pyIfPart : ((PyIfStatement) element).getElifParts()) {
        executableLines.add(new PythonTokenLocation(pyIfPart).startLine());
      }
    } else if (element instanceof PyTryExceptStatement) {
      for (PyExceptPart pyExceptPart : ((PyTryExceptStatement) element).getExceptParts()) {
        executableLines.add(new PythonTokenLocation(pyExceptPart).startLine());
      }
    }
  }

  private void handleComment(PsiComment comment) {
    String[] commentLines = getCommentContent(comment.getText())
      .split("(\r)?\n|\r", -1);
    int line = new PythonTokenLocation(comment).startLine();

    for (String commentLine : commentLines) {
      if (commentLine.contains("NOSONAR")) {
        noSonar.add(line);
      } else if (!isBlankComment(commentLine) && !noSonar.contains(line)) {
        linesOfComments.add(line);
      }
      line++;
    }
  }

  @Override
  public void visitPyStringLiteralExpression(PyStringLiteralExpression node) {
    if (node.isDocString()) {
      PythonTokenLocation location = new PythonTokenLocation(node);
      for (int line = location.startLine(); line <= location.endLine(); line++) {
        linesOfComments.add(line);
      }
    }
    super.visitPyStringLiteralExpression(node);
  }

  private static boolean isDocString(PyStatement element) {
    if (element instanceof PyExpressionStatement) {
      PyExpression expression = ((PyExpressionStatement) element).getExpression();
      return expression instanceof PyStringLiteralExpression
        && ((PyStringLiteralExpression) expression).isDocString();
    }
    return false;
  }

  private static boolean isBlankComment(String line) {
    for (int i = 0; i < line.length(); i++) {
      if (Character.isLetterOrDigit(line.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  private static String getCommentContent(String comment) {
    // Comment always starts with "#"
    return comment.substring(comment.indexOf('#'));
  }

  public Set<Integer> getLinesWithNoSonar() {
    return Collections.unmodifiableSet(noSonar);
  }

  public Set<Integer> getExecutableLines() {
    return Collections.unmodifiableSet(executableLines);
  }

  public Set<Integer> getLinesOfCode() {
    return Collections.unmodifiableSet(linesOfCode);
  }

  public int getCommentLineCount() {
    return linesOfComments.size();
  }

}
