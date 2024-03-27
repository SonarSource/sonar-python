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

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import java.lang.reflect.Type;
import java.util.Arrays;

public class PolymorphDeserializer<T> implements JsonDeserializer<T> {
  @Override
  public T deserialize(JsonElement json, Type type, JsonDeserializationContext context) throws JsonParseException {
    try {
      Class<?> typeClass = Class.forName(type.getTypeName());
      JsonType jsonType = typeClass.getDeclaredAnnotation(JsonType.class);
      String property = json.getAsJsonObject().get(jsonType.property()).getAsString();
      JsonSubtype[] subtypes = jsonType.subtypes();
      Type subType = Arrays.stream(subtypes).filter(subtype -> subtype.name().equals(property)).findFirst().orElseThrow(() -> new IllegalArgumentException("Unknwon type:" + property)).child();
      return context.deserialize(json, subType);
    } catch (Exception e) {
      throw new JsonParseException("Failed deserialize json", e);
    }
  }
}
