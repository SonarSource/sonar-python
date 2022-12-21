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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.python.TokenLocation;

public class CpdSerializer {

  private CpdSerializer() {
    // Prevent instantiation
  }

  public static final class TokenInfo implements Serializable {
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
    List<TokenInfo> tokenInfos = tokens.stream()
      .map(TokenInfo::from)
      .collect(Collectors.toList());

    ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
    ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);
    objectOutputStream.writeObject(tokenInfos);

    return byteArrayOutputStream.toByteArray();
  }

  public static List<TokenInfo> fromBytes(byte[] bytes) throws IOException, ClassNotFoundException {
    ObjectInputStream objectInputStream = new ObjectInputStream(new ByteArrayInputStream(bytes));
    return (List<TokenInfo>) objectInputStream.readObject();
  }
}
