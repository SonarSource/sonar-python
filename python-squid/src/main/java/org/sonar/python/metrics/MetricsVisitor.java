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

import com.intellij.psi.PsiElement;
import com.jetbrains.python.psi.PyExceptPart;
import com.jetbrains.python.psi.PyExpression;
import com.jetbrains.python.psi.PyExpressionStatement;
import com.jetbrains.python.psi.PyIfPart;
import com.jetbrains.python.psi.PyIfStatement;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import com.jetbrains.python.psi.PyStatement;
import com.jetbrains.python.psi.PyStringLiteralExpression;
import com.jetbrains.python.psi.PyTryExceptStatement;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.python.frontend.PythonTokenLocation;

/**
 * Visitor that computes {@link CoreMetrics#NCLOC_DATA_KEY} and {@link CoreMetrics#COMMENT_LINES} metrics used by the DevCockpit.
 */
public class MetricsVisitor extends PyRecursiveElementVisitor {

  private final boolean ignoreHeaderComments;
  private Set<Integer> executableLines = new HashSet<>();

  public MetricsVisitor(boolean ignoreHeaderComments) {
    this.ignoreHeaderComments = ignoreHeaderComments;
  }

  @Override
  public void visitElement(PsiElement element) {
    if (element instanceof PyStatement) {
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
    super.visitElement(element);
  }

  private static boolean isDocString(PsiElement element) {
    if (element instanceof PyExpressionStatement) {
      PyExpression expression = ((PyExpressionStatement) element).getExpression();
      return expression instanceof PyStringLiteralExpression
        && ((PyStringLiteralExpression) expression).isDocString();
    }
    return false;
  }

  public Set<Integer> getExecutableLines() {
    return Collections.unmodifiableSet(new HashSet<>(executableLines));
  }
}
