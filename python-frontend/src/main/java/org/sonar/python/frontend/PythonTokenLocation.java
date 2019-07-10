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
package org.sonar.python.frontend;

import com.intellij.openapi.editor.Document;
import com.intellij.psi.PsiElement;
import org.jetbrains.annotations.NotNull;

public class PythonTokenLocation {
  private final int startLine;
  private final int startLineOffset;
  private final int endLine;
  private final int endLineOffset;

  public PythonTokenLocation(@NotNull PsiElement element) {
    Document psiDocument = element.getContainingFile().getViewProvider().getDocument();
    int startOffset = element.getTextRange().getStartOffset();
    startLine = psiDocument.getLineNumber(startOffset);
    int startLineNumberOffset = psiDocument.getLineStartOffset(startLine);
    startLineOffset = startOffset - startLineNumberOffset;
    int endOffset = element.getTextRange().getEndOffset();
    endLine = psiDocument.getLineNumber(endOffset);
    int endLineNumberOffset = psiDocument.getLineStartOffset(endLine);
    endLineOffset = endOffset - endLineNumberOffset;
  }

  public int startLine() {
    return startLine + 1;
  }

  public int startLineOffset() {
    return startLineOffset;
  }

  public int endLine() {
    return endLine + 1;
  }

  public int endLineOffset() {
    return endLineOffset;
  }
}
