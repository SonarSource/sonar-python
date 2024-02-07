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

import com.google.common.annotations.VisibleForTesting;
import com.google.gson.Gson;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;

public class TypeContext {
  private static final class JsonTypeInfo {
    String text;
    int start_line;
    int start_col;
    String syntax_role;
    String type;
    String short_type;

    @Override
    public String toString() {
      return "TypeInfo{" +
        "text='" + text + '\'' +
        ", start_line=" + start_line +
        ", start_col=" + start_col +
        ", syntax_role='" + syntax_role + '\'' +
        ", type='" + type + '\'' +
        ", short_type='" + short_type + '\'' +
        '}';
    }
  }

  private static final class NameAccess {
    final String fileName;
    final int line;
    final int column;
    final String name;

    public NameAccess(String fileName, int line, int column, String name) {
      this.fileName = fileName;
      this.line = line;
      this.column = column;
      this.name = name;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o)
        return true;
      if (o == null || getClass() != o.getClass())
        return false;
      NameAccess that = (NameAccess) o;
      return line == that.line && column == that.column && Objects.equals(fileName, that.fileName) && Objects.equals(name, that.name);
    }

    @Override
    public int hashCode() {
      return Objects.hash(fileName, line, column, name);
    }
  }

  private static final Logger LOG = LoggerFactory.getLogger(TypeContext.class);
  private final Map<String, List<JsonTypeInfo>> files;

  private Map<NameAccess, JsonTypeInfo> typesByPosition = null;

  public TypeContext() {
    this.files = new HashMap<>();
  }

  public static TypeContext fromJSON(String json) {
    return new Gson().fromJson("{files: " + json + "}", TypeContext.class);
  }

  private void populateTypesByPosition() {
    typesByPosition = new HashMap<>();
    files.forEach((file, typeInfos) -> {
      for (JsonTypeInfo typeInfo : typeInfos) {
        typesByPosition.put(new NameAccess(file, typeInfo.start_line, typeInfo.start_col, typeInfo.text), typeInfo);
      }
    });
  }

  @VisibleForTesting
  Optional<InferredType> getTypeFor(String fileName, int line, int column, String name, String kind) {
    if (typesByPosition == null) {
      populateTypesByPosition();
    }
    NameAccess nameAccess = new NameAccess(fileName, line, column, name);
    if (!typesByPosition.containsKey(nameAccess))
      return Optional.empty();
    JsonTypeInfo typeInfo = typesByPosition.get(nameAccess);
    if (!typeInfo.syntax_role.equals(kind)) {
      LOG.error("Found type at position, but does not match expected kind ({}): {}", kind, typeInfo);
      return Optional.empty();
    }
    return Optional.of(typeStringToTypeInfo(typeInfo.short_type, typeInfo.type, fileName));
  }

  private static InferredType typeStringToTypeInfo(String typeString, String detailedType, String fileName) {
    if ("None".equals(typeString)) {
      typeString = "NoneType";
    }
    // workaround until Typeshed is fixed
    if (detailedType.startsWith("builtins.") ||
      detailedType.startsWith("GenericType(base_type=ClassType(builtins.") ||
      detailedType.startsWith("TupleType(base_type=ClassType(builtins.")) {
      return InferredTypes.runtimeBuiltinType(getBaseType(typeString));
    } else if (typeString.indexOf('.') >= 0 || !detailedType.startsWith(typeString)) {
      ClassSymbolImpl callableClassSymbol = new ClassSymbolImpl(typeString, detailedType);
      return new RuntimeType(callableClassSymbol);
    } else if (typeString.equals("Any")) {
      return InferredTypes.anyType();
    } else {
      // Try to make a qualified name. pytype does not give a qualified name for classes defined in the same file.
      // The filename should contain the module path, but we may get and extra prefix. The prefix will get removed later.
      var qualifiedTypeString = fileName.replace('/', '.').substring(0, fileName.lastIndexOf('.') + 1) + typeString;
      ClassSymbolImpl callableClassSymbol = new ClassSymbolImpl(typeString, qualifiedTypeString);
      return new RuntimeType(callableClassSymbol);
    }
  }

  private static String getBaseType(String typeString) { // Tuple[int, int]
    if (typeString.startsWith("Tuple")) {
      return "tuple";
    } else if (typeString.startsWith("List")) {
      return "list";
    } else if (typeString.startsWith("Set")) {
      return "set";
    } else if (typeString.startsWith("Dict")) {
      return "dict";
    } else if (typeString.startsWith("Type")) {
      return "type";
    } else {
      return typeString;
    }
  }

  public Optional<InferredType> getTypeFor(String fileName, Name name) {
    Token token = name.firstToken();
    return getTypeFor(fileName, token.line(), token.column(), name.name(), "Variable");
  }

  public Optional<InferredType> getTypeFor(String fileName, QualifiedExpression attributeAccess) {
    Token token = attributeAccess.firstToken();
    return getTypeFor(fileName, token.line(), token.column(), attributeAccess.name().name(), "Attribute");
  }
}
