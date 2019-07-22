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
import com.intellij.psi.tree.IElementType;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.PyTokenTypes;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class ComplexityVisitor extends PyRecursiveElementVisitor {

  private int complexity;

  public static int complexity(PsiElement element) {
    ComplexityVisitor visitor = isFunctionDeclaration(element) ? new FunctionComplexityVisitor() : new ComplexityVisitor();
    element.accept(visitor);
    return visitor.complexity;
  }

  private static boolean isFunctionDeclaration(PsiElement element) {
    return PyElementTypes.FUNCTION_DECLARATION.equals(element.getNode().getElementType());
  }

  private static final Set<IElementType> COMPLEXITY_TYPES = new HashSet<>(Arrays.asList(
    PyElementTypes.FUNCTION_DECLARATION,
    PyElementTypes.FOR_STATEMENT,
    PyElementTypes.WHILE_STATEMENT,
    PyTokenTypes.IF_KEYWORD,
    PyTokenTypes.AND_KEYWORD,
    PyTokenTypes.OR_KEYWORD
  ));

  @Override
  public void visitElement(PsiElement element) {
    if (COMPLEXITY_TYPES.contains(element.getNode().getElementType())) {
      complexity++;
    }
    super.visitElement(element);
  }

  public int getComplexity() {
    return complexity;
  }

  private static class FunctionComplexityVisitor extends ComplexityVisitor {

    private int functionNestingLevel = 0;

    @Override
    public void visitElement(PsiElement element) {
      if (isFunctionDeclaration(element)) {
        functionNestingLevel++;
      }
      if (functionNestingLevel == 1) {
        super.visitElement(element);
      }
      if (isFunctionDeclaration(element)) {
        functionNestingLevel--;
      }
    }
  }

}
