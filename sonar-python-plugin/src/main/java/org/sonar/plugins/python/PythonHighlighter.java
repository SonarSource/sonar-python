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
package org.sonar.plugins.python;

import com.intellij.psi.PsiElement;
import com.intellij.psi.PsiWhiteSpace;
import com.intellij.psi.impl.source.tree.LeafPsiElement;
import com.jetbrains.python.PyTokenTypes;
import com.jetbrains.python.psi.PyElementType;
import com.jetbrains.python.psi.PyRecursiveElementVisitor;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.highlighting.NewHighlighting;
import org.sonar.api.batch.sensor.highlighting.TypeOfText;
import org.sonar.python.frontend.PythonKeyword;
import org.sonar.python.frontend.PythonTokenLocation;

/**
 * Colors Python code. Currently colors:
 * <ul>
 * <li>
 * String literals. Examples:
 * <pre>
 *       "hello"
 *
 *       'hello'
 *
 *       """ hello
 *           hello again
 *       """
 *     </pre>
 * </li>
 * <li>
 * Keywords. Example:
 * <pre>
 *       def
 *     </pre>
 * </li>
 * <li>
 * Numbers. Example:
 * <pre>
 *        123
 *        123L
 *        123.45
 *        123.45e-10
 *        123+88.99J
 *     </pre>
 * For a negative number, the "minus" sign is not colored.
 * </li>
 * <li>
 * Comments. Example:
 * <pre>
 *        # some comment
 *     </pre>
 * </li>
 * </ul>
 * Docstrings are handled (i.e., colored) as structured comments, not as normal string literals.
 * "Attribute docstrings" and "additional docstrings" (see PEP 258) are handled as normal string literals.
 * Reminder: a docstring is a string literal that occurs as the first statement in a module,
 * function, class, or method definition.
 */
public class PythonHighlighter extends PyRecursiveElementVisitor {

  private NewHighlighting newHighlighting;

  PythonHighlighter(SensorContext context, InputFile inputFile) {
    newHighlighting = context.newHighlighting();
    newHighlighting.onFile(inputFile);
  }

  NewHighlighting getNewHighlighting() {
    return newHighlighting;
  }

  @Override
  public void visitElement(PsiElement element) {
    if (element instanceof LeafPsiElement && !(element instanceof PsiWhiteSpace)) {
      LeafPsiElement leaf = (LeafPsiElement) element;
      PyElementType elementType = (PyElementType) leaf.getElementType();
      if (PythonKeyword.isKeyword(elementType)) {
        highlight(leaf, TypeOfText.KEYWORD);
      } else if (PyTokenTypes.NUMERIC_LITERALS.contains(elementType)) {
        highlight(leaf, TypeOfText.CONSTANT);
      } else if (elementType == PyTokenTypes.DOCSTRING) {
        highlight(leaf, TypeOfText.STRUCTURED_COMMENT);
      } else if (PyTokenTypes.STRING_NODES.contains(elementType)) {
        highlight(leaf, TypeOfText.STRING);
      } else if (elementType == PyTokenTypes.END_OF_LINE_COMMENT) {
        highlight(leaf, TypeOfText.COMMENT);
      }
    }
    super.visitElement(element);
  }

  private void highlight(PsiElement token, TypeOfText typeOfText) {
    PythonTokenLocation tokenLocation = new PythonTokenLocation(token);
    newHighlighting.highlight(tokenLocation.startLine(), tokenLocation.startLineOffset(), tokenLocation.endLine(), tokenLocation.endLineOffset(), typeOfText);
  }

}
