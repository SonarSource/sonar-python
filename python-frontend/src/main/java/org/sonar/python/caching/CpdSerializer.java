/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.python.caching;

import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.python.TokenLocation;
import org.sonar.python.types.protobuf.CpdTokenProtos;

public class CpdSerializer {

  private CpdSerializer() {
    // Prevent instantiation
  }

  public static final class TokenInfo {
    public final int startLine;
    public final int startLineOffset;
    public final int endLine;
    public final int endLineOffset;
    public final String value;

    public static TokenInfo from(Token token) {
      TokenLocation location = new TokenLocation(token);
      return new TokenInfo(location.startLine(), location.startLineOffset(), location.endLine(), location.endLineOffset(), token.value());
    }

    public TokenInfo(int startLine, int startLineOffset, int endLine, int endLineOffset, String value) {
      this.startLine = startLine;
      this.startLineOffset = startLineOffset;
      this.endLine = endLine;
      this.endLineOffset = endLineOffset;
      this.value = value;
    }
  }

  public static byte[] toBytes(List<Token> tokens) throws IOException {
    CpdTokenProtos.FileCpdTokens.Builder builder = CpdTokenProtos.FileCpdTokens.newBuilder();

    for (Token token : tokens) {
      TokenLocation location = new TokenLocation(token);
      CpdTokenProtos.Token protoToken = CpdTokenProtos.Token.newBuilder()
        .setValue(token.value())
        .setStartLine(location.startLine())
        .setStartLineOffset(location.startLineOffset())
        .setEndLine(location.endLine())
        .setEndLineOffset(location.endLineOffset())
        .build();

      builder.addTokens(protoToken);
    }

    return builder.build().toByteArray();
  }

  public static List<TokenInfo> fromBytes(byte[] bytes) throws IOException {
    return CpdTokenProtos.FileCpdTokens.parseFrom(bytes)
      .getTokensList()
      .stream()
      .map(proto -> new TokenInfo(proto.getStartLine(), proto.getStartLineOffset(), proto.getEndLine(), proto.getEndLineOffset(), proto.getValue()))
      .collect(Collectors.toList());
  }
}
