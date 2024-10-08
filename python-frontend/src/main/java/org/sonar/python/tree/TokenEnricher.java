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
      int startCol = token.getColumn();
      int endCol = token.getColumn() + token.getValue().length();
      int ipynbStartCol = computeColWithEscapes(location.column(), startCol, escapeCharsMap);
      int escapedCharInToken = computeEscapeCharsInToken(escapeCharsMap, startCol, endCol);
      List<Trivia> trivia = token.getTrivia().stream()
        .map(t -> computeTriviaLocation(t, location.line(), ipynbStartCol, token.getLine(), offsetMap))
        .toList();

      return new TokenImpl(token, location.line(), ipynbStartCol, escapedCharInToken, trivia, location.isCompresssed());
    }
    return new TokenImpl(token);
  }

  private static Trivia computeTriviaLocation(com.sonar.sslr.api.Trivia trivia, int parentLine, int parentCol, int parentPythonLine,
    Map<Integer, IPythonLocation> offsetMap) {
    var line = parentLine;
    int escapedCharInToken = computeEscapeCharsInTrivia(trivia, offsetMap);
    var col = parentCol - escapedCharInToken - trivia.getToken().getValue().length();
    var isCompressed = false;
    if (parentPythonLine != trivia.getToken().getLine()) {
      IPythonLocation location = offsetMap.get(trivia.getToken().getLine());
      line = location.line();
      Map<Integer, Integer> escapeCharsMap = location.colOffset();
      col = computeColWithEscapes(location.column(), trivia.getToken().getColumn(), escapeCharsMap);
      isCompressed = location.isCompresssed();
    }
    return new TriviaImpl(new TokenImpl(trivia.getToken(), line, col,
      escapedCharInToken, List.of(), isCompressed));
  }

  private static int computeColWithEscapes(int offsetColumn, int currentCol, Map<Integer, Integer> escapes) {
    int escapedCharsOffset = computeEscapeCharsInToken(escapes, 0, currentCol);
    return offsetColumn + currentCol + escapedCharsOffset;
  }

  private static int computeEscapeCharsInTrivia(com.sonar.sslr.api.Trivia trivia, Map<Integer, IPythonLocation> offsetMap) {
    IPythonLocation location = offsetMap.get(trivia.getToken().getLine());
    Token token = trivia.getToken();
    int startCol = token.getColumn();
    int endCol = token.getColumn() + token.getValue().length();
    return computeEscapeCharsInToken(location.colOffset(), startCol, endCol);
  }

  private static int computeEscapeCharsInToken(Map<Integer, Integer> escapedMap, int startCol, int endCol) {
    return escapedMap.entrySet().stream()
      .filter(entry -> entry.getKey() >= startCol && entry.getKey() < endCol)
      .mapToInt(Map.Entry::getValue)
      .sum();
  }

}
