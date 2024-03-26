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
package org.sonar.python.types.pytype.json;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.sonar.python.types.PyTypeDetailedInfo;
import org.sonar.python.types.PyTypeDetailedInfoDeserializer;
import org.sonar.python.types.pytype.BaseType;
import org.sonar.python.types.pytype.PyTypeInfo;
import org.sonar.python.types.pytype.PyTypeTable;

public class PyTypeTableReader {

  private final Gson gson;
  private final Type type;

  public static PyTypeTable fromJsonString(String json) {
    try (var reader = new StringReader(json)) {
      return new PyTypeTableReader().fromJson(reader);
    }
  }

  public static PyTypeTable fromJsonPath(Path path) {
    try {
      return new PyTypeTableReader().fromJson(path);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public PyTypeTableReader() {
    gson = new GsonBuilder()
      .registerTypeAdapter(PyTypeDetailedInfo.class, new PyTypeDetailedInfoDeserializer())
      .registerTypeAdapter(BaseType.class, new PolymorphDeserializer<>())
      .create();
    type = new TypeToken<Map<String, List<PyTypeInfo>>>() {
    }.getType();
  }

  public PyTypeTable fromJson(Path path) throws IOException {
    try (var reader = Files.newBufferedReader(path)) {
      return fromJson(reader);
    }
  }

  private PyTypeTable fromJson(Reader reader) {
    var files = gson.<Map<String, List<PyTypeInfo>>>fromJson(reader, type);
    return new PyTypeTable(files);
  }

}
