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

import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import com.sonar.sslr.api.TokenType;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.sonar.python.IPythonLocation;
import org.sonar.python.api.PythonTokenType;

public class TokenEnricher {

  private static final Set<Character> ESCAPED_CHARS = Set.of('"', '\'', '\\', '\b', '\f', '\n', '\r', '\t');
  private static final Set<TokenType> TOKEN_TYPES_TO_IGNORE = Set.of(PythonTokenType.DEDENT, GenericTokenType.EOF, PythonTokenType.INDENT);

  private TokenEnricher() {
  }

  public static List<TokenImpl> enrichTokens(List<Token> tokens, Map<Integer, IPythonLocation> offsetMap) {
    return tokens.stream().map(token -> enrichToken(token, offsetMap)).toList();
  }

  public static TokenImpl enrichToken(Token token, Map<Integer, IPythonLocation> offsetMap) {
    if (!offsetMap.isEmpty() && !TOKEN_TYPES_TO_IGNORE.contains(token.getType())) {
      IPythonLocation location = offsetMap.get(token.getLine());
      if (location == null) {
        throw new IllegalStateException(String.format("No IPythonLocation found for line %s", token.getLine()));
      }
      Map<Integer, Integer> escapeCharsMap = location.colOffset();
      int startCol = computeColWithEscapes(token.getColumn(), escapeCharsMap, location.column());
      int escapedCharInToken = 0;
      for (int i = 0; i < token.getValue().length(); i++) {
        if (ESCAPED_CHARS.contains(token.getValue().charAt(i))) {
          escapedCharInToken++;
        }
      }
      return new TokenImpl(token, location.line(), startCol, escapedCharInToken);
    }
    return new TokenImpl(token);
  }

  private static int computeColWithEscapes(int currentCol, Map<Integer, Integer> escapes, int offsetColumn) {
    return (int) escapes.keySet().stream().filter(k -> k < currentCol).count() + offsetColumn + currentCol;
  }

}
