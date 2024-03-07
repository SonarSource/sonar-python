/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.types;

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

public class TypeContextReader {

  private final Gson gson;
  private final Type type;

  public static TypeContext fromJson(String json) {
    try (var reader = new StringReader(json)) {
      return new TypeContextReader().fromJson(reader);
    }
  }

  public TypeContextReader() {
    gson = new GsonBuilder()
      .registerTypeAdapter(PyTypeDetailedInfo.class, new PyTypeDetailedInfoDeserializer())
      .create();
    type = new TypeToken<Map<String, List<PyTypeInfo>>>() {
    }.getType();
  }

  public TypeContext fromJson(Path path) throws IOException {
    try (var reader = Files.newBufferedReader(path)) {
      return fromJson(reader);
    }
  }

  private TypeContext fromJson(Reader reader) {
    var files = gson.<Map<String, List<PyTypeInfo>>>fromJson(reader, type);
    return new TypeContext(files);
  }

}
