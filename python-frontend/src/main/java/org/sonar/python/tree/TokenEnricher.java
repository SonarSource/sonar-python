/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.tree;

import com.sonar.sslr.api.Token;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonar.python.IPythonLocation;

public class TokenEnricher {

  private static final Set<Character> ESCAPED_CHARS = Set.of('"', '\\', '\b', '\f', '\n', '\r', '\t');

  private TokenEnricher() {
  }

  public static List<TokenImpl> enrichTokens(List<Token> tokens, Map<Integer, IPythonLocation> offsetMap) {
    return tokens.stream().map(token -> enrichToken(token, offsetMap)).toList();
  }

  public static TokenImpl enrichToken(Token token, Map<Integer, IPythonLocation> offsetMap) {
    if (!offsetMap.isEmpty()) {
      IPythonLocation location = offsetMap.get(token.getLine());
      if (location == null) {
        throw new IllegalStateException(String.format("No IPythonLocation found for line %s", token.getLine()));
      }
      Map<Integer, Integer> escapeCharsMap = location.colOffset();
      int startCol = computeColWithEscapes(token.getColumn(), escapeCharsMap, location.column());
      int escapedCharInToken = computeEscapeCharsInToken(token.getValue());
      List<Trivia> trivia = token.getTrivia().stream()
        .map(t -> computeTriviaLocation(t, location.line(), startCol, token.getLine(), offsetMap))
        .toList();

      return new TokenImpl(token, location.line(), startCol, escapedCharInToken, trivia);
    }
    return new TokenImpl(token);
  }

  private static Trivia computeTriviaLocation(com.sonar.sslr.api.Trivia trivia, int parentLine, int parentCol, int parentPythonLine, Map<Integer, IPythonLocation> offsetMap) {
    int escapedCharInToken = computeEscapeCharsInToken(trivia.getToken().getValue());
    var line = parentLine;
    var col = parentCol - escapedCharInToken - trivia.getToken().getValue().length();
    if (parentPythonLine != trivia.getToken().getLine()) {
      IPythonLocation location = offsetMap.get(trivia.getToken().getLine());
      line = location.line();
      Map<Integer, Integer> escapeCharsMap = location.colOffset();
      col = computeColWithEscapes(trivia.getToken().getColumn(), escapeCharsMap, location.column());
    }
    return new TriviaImpl(new TokenImpl(trivia.getToken(), line, col,
      escapedCharInToken, List.of()));
  }

  private static int computeEscapeCharsInToken(String tokenValue) {
    int escapedCharInToken = 0;
    for (int i = 0; i < tokenValue.length(); i++) {
      if (ESCAPED_CHARS.contains(tokenValue.charAt(i))) {
        escapedCharInToken++;
      }
    }
    return escapedCharInToken;

  }

  private static int computeColWithEscapes(int currentCol, Map<Integer, Integer> escapes, int offsetColumn) {
    return (int) escapes.keySet().stream().filter(k -> k > 0 && k < currentCol).count() + offsetColumn + currentCol;
  }

}
