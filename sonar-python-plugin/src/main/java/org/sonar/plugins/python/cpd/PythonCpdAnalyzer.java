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
package org.sonar.plugins.python.cpd;

import com.intellij.openapi.editor.Document;
import com.intellij.psi.PsiElement;
import com.intellij.psi.tree.IElementType;
import com.jetbrains.python.PyTokenTypes;
import com.jetbrains.python.lexer.PythonIndentingLexer;
import com.jetbrains.python.psi.PyElementType;
import com.jetbrains.python.psi.PyFile;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.cpd.NewCpdTokens;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.python.frontend.PythonParser;
import org.sonar.python.frontend.PythonTokenLocation;

public class PythonCpdAnalyzer {

  private final SensorContext context;
  private static final Set<PyElementType> IGNORED_TOKEN_TYPES = new HashSet<>(Arrays.asList(
    PyTokenTypes.LINE_BREAK, PyTokenTypes.DEDENT, PyTokenTypes.INDENT, PyTokenTypes.END_OF_LINE_COMMENT, PyTokenTypes.SPACE, PyTokenTypes.STATEMENT_BREAK));
  private static final Logger LOG = Loggers.get(PythonCpdAnalyzer.class);

  public PythonCpdAnalyzer(SensorContext context) {
    this.context = context;
  }

  public void pushCpdTokens(InputFile inputFile, PyFile pyFile, String fileContent) {
    Document document = getDocument(pyFile);
    if (document == null) {
      LOG.debug("Cannot complete CPD analysis: PSIDocument is null.");
      return;
    }
    PythonIndentingLexer lexer = new PythonIndentingLexer();
    lexer.start(PythonParser.normalizeEol(fileContent));
    NewCpdTokens cpdTokens = context.newCpdTokens().onFile(inputFile);
    IElementType prevTokenType = null;
    while (lexer.getTokenType() != null) {
      IElementType currentTokenType = lexer.getTokenType();
      // INDENT/DEDENT could not be completely ignored during CPD see https://docs.python.org/3/reference/lexical_analysis.html#indentation
      // Just taking into account DEDENT is enough, but because the DEDENT token has an empty value, it's the
      // following new line which is added in its place to create a difference
      if (isNewLineWithIndentationChange(prevTokenType, currentTokenType) || !IGNORED_TOKEN_TYPES.contains(currentTokenType)) {
        int tokenEnd = lexer.getTokenEnd();
        String tokenText = lexer.getTokenText();
        if (currentTokenType == PyTokenTypes.LINE_BREAK) {
          tokenText = "\n";
          tokenEnd = lexer.getTokenStart() + 1;
        }
        PythonTokenLocation location = new PythonTokenLocation(lexer.getTokenStart(), tokenEnd, document);
        cpdTokens.addToken(location.startLine(), location.startLineOffset(), location.endLine(), location.endLineOffset(), tokenText);
      }
      prevTokenType = currentTokenType;
      lexer.advance();
    }

    cpdTokens.save();
  }

  private static boolean isNewLineWithIndentationChange(@CheckForNull IElementType prevTokenType, IElementType currentTokenType) {
    return prevTokenType != null && prevTokenType == PyTokenTypes.DEDENT && currentTokenType == PyTokenTypes.LINE_BREAK;
  }

  @CheckForNull
  private static Document getDocument(PyFile pyFile) {
    PsiElement root = pyFile.getFirstChild();
    if (root == null) {
      return null;
    }
    return root.getContainingFile().getViewProvider().getDocument();
  }

}
