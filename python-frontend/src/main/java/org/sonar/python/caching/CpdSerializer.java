/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.caching;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.python.TokenLocation;

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

  public static class SerializationResult {
    public final byte[] data;
    public final byte[] stringTable;

    public SerializationResult(byte[] data, byte[] stringTable) {
      this.data = data;
      this.stringTable = stringTable;
    }
  }

  public static SerializationResult serialize(List<Token> tokens) throws IOException {
    return new Serializer().convert(tokens);
  }

  public static List<TokenInfo> deserialize(byte[] dataBytes, byte[] stringTableBytes) throws IOException {
    return new Deserializer(new VarLengthInputStream(dataBytes), new VarLengthInputStream(stringTableBytes)).convert();
  }

  private static class Serializer {
    private final ByteArrayOutputStream stream;
    private final VarLengthOutputStream out;
    private final StringTable stringTable;

    private Serializer() {
      stream = new ByteArrayOutputStream();
      out = new VarLengthOutputStream(stream);
      stringTable = new StringTable();
    }

    public SerializationResult convert(List<Token> tokens) throws IOException {
      try (out; stream) {
        writeInt(tokens.size());
        for (Token token : tokens) {
          write(token);
        }
        out.writeUTF("END");

        return new SerializationResult(stream.toByteArray(), writeStringTable());
      }
    }

    private void write(Token token) throws IOException {
      TokenLocation location = new TokenLocation(token);
      writeInt(location.startLine());
      writeInt(location.startLineOffset());
      writeInt(location.endLine());
      writeInt(location.endLineOffset());
      writeText(token.value());
    }

    private void writeText(@Nullable String text) throws IOException {
      out.writeInt(stringTable.getIndex(text));
    }

    private void writeInt(int number) throws IOException {
      out.writeInt(number);
    }

    private byte[] writeStringTable() throws IOException {
      ByteArrayOutputStream stringTableStream = new ByteArrayOutputStream();
      VarLengthOutputStream output = new VarLengthOutputStream(stringTableStream);
      List<String> byIndex = stringTable.getStringList();
      output.writeInt(byIndex.size());
      for (String string : byIndex) {
        output.writeUTF(string);
      }

      output.writeUTF("END");
      return stringTableStream.toByteArray();
    }
  }

  private static class Deserializer {
    private final VarLengthInputStream in;
    private final VarLengthInputStream stringTableIn;

    private StringTable stringTable;

    private Deserializer(VarLengthInputStream in, VarLengthInputStream stringTableIn) {
      this.in = in;
      this.stringTableIn = stringTableIn;
    }

    public List<TokenInfo> convert() throws IOException {
      try (in; stringTableIn) {
        stringTable = readStringTable();
        int sizeOfCpdTokens = readInt();
        List<TokenInfo> cpdTokens = new ArrayList<>(sizeOfCpdTokens);

        for (int i = 0; i < sizeOfCpdTokens; i++) {
          readCpdToken(cpdTokens);
        }

        if (!"END".equals(in.readUTF())) {
          throw new IOException("Can't read data from cache, format corrupted");
        }
        return cpdTokens;
      }
    }

    private void readCpdToken(List<TokenInfo> cpdTokens) throws IOException {
      cpdTokens.add(new TokenInfo(
        readInt(),
        readInt(),
        readInt(),
        readInt(),
        readString()
      ));
    }

    private int readInt() throws IOException {
      return in.readInt();
    }

    private String readString() throws IOException {
      return stringTable.getString(in.readInt());
    }

    private StringTable readStringTable() throws IOException {
      int size = stringTableIn.readInt();
      List<String> byIndex = new ArrayList<>(size);
      for (int i = 0; i < size; i++) {
        byIndex.add(stringTableIn.readUTF());
      }
      if (!"END".equals(stringTableIn.readUTF())) {
        throw new IOException("Can't read data from cache, format corrupted");
      }
      return new StringTable(byIndex);
    }
  }

}
