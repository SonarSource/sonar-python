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
package org.sonar.python.types;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.antlr.v4.runtime.RecognitionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

public class PyTypeTypeGrammar {

  static ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
  static String fileName = "";

  private static final Logger LOG = LoggerFactory.getLogger(PyTypeTypeGrammar.class);

  private static final InferredType iterator = new RuntimeType(new ClassSymbolImpl("Iterator", "typing.Iterator"));
  // complete list https://github.com/google/pytype/blob/95cb0b760cf9a5b8d7d7b315b4440d2e56519072/pytype/pytd/abc_hierarchy.py#L60
  private static final Map<String, InferredType> builtinsTranslation = Map.ofEntries(
    Map.entry("generator", (InferredType) new RuntimeType(new ClassSymbolImpl("Generator", "typing.Generator"))),
    Map.entry("module", (InferredType) new RuntimeType(new ClassSymbolImpl("ModuleType", "types.ModuleType"))),
    Map.entry("listiterator", iterator),
    Map.entry("bytearray_iterator", iterator),
    Map.entry("dict_keys", InferredTypes.anyType()),
    Map.entry("dict_items", InferredTypes.anyType()),
    Map.entry("dict_values", InferredTypes.anyType()),
    Map.entry("dict_keyiterator", iterator),
    Map.entry("dict_valueiterator", iterator),
    Map.entry("dict_itemiterator", iterator),
    Map.entry("list_iterator", iterator),
    Map.entry("list_reverseiterator", iterator),
    Map.entry("range_iterator", iterator),
    Map.entry("longrange_iterator", iterator),
    Map.entry("set_iterator", iterator),
    Map.entry("tuple_iterator", iterator),
    Map.entry("str_iterator", iterator),
    Map.entry("zip_iterator", iterator),
    Map.entry("bytes_iterator", iterator),
    Map.entry("mappingproxy", (InferredType) new RuntimeType(new ClassSymbolImpl("Mapping", "typing.Mapping"))),
    Map.entry("async_generator", (InferredType) new RuntimeType(new ClassSymbolImpl("Generator", "typing.AsyncGenerator"))),
    Map.entry("coroutine", (InferredType) new RuntimeType(new ClassSymbolImpl("Coroutine", "types.Coroutine"))),
    Map.entry("code", (InferredType) new RuntimeType(new ClassSymbolImpl("CodeType", "types.CodeType"))));

  private static final Pattern builtinsPattern = Pattern.compile("^(?:ClassType\\()?builtins\\.(\\w+)(?:\\))?");
  private static final Pattern classTypePattern = Pattern.compile("^ClassType\\(([\\w\\.]+)\\)$");
  private static final Pattern anyTypePattern = Pattern.compile("^(Any|No)thingType\\(\\)$");
  private static final Pattern unionTypePattern = Pattern.compile("^UnionType\\(type_list=\\((.+)\\)\\)$");
  private static final Pattern genericTypePatter = Pattern.compile("^GenericType\\(base_type=(.+), parameters=\\((.+),\\)\\)$");
  private static final Pattern namedTypePattern = Pattern.compile("(\\w+(\\.\\w+)?)");

  public static InferredType getTypeFromString(String detailedType) {
    Matcher m = builtinsPattern.matcher(detailedType);
    if (m.find()) {
      String typeName = m.group(1);
      if (builtinsTranslation.containsKey(typeName)) {
        return builtinsTranslation.get(typeName);
      }
      return InferredTypes.runtimeBuiltinType(typeName);
    }
    m = classTypePattern.matcher(detailedType);
    if (m.find()) {
      String typeName = m.group(1);
      return getInferredType(typeName);
    }
    m = unionTypePattern.matcher(detailedType);
    if (m.find()) {
      var typeList = m.group(1);
      var innerTypeNames = parseTopLevelTypeString(typeList);
      Stream<InferredType> innerTypes = innerTypeNames.stream().map(PyTypeTypeGrammar::getTypeFromString);
      return InferredTypes.union(innerTypes);
    }
    m = genericTypePatter.matcher(detailedType);
    if (m.find()) {
      var baseType = m.group(1);
      if(baseType.equals("ClassType(builtins.type)")){
        var parameter = m.group(2);
        var firstParameterType = parameter.split(",")[0].trim();
        return getTypeFromString(firstParameterType);
      }
      return getTypeFromString(baseType);
    }
    m = anyTypePattern.matcher(detailedType);
    if (m.matches()) {
      return InferredTypes.anyType();
    }
    m = namedTypePattern.matcher(detailedType);
    if (m.find()) {
      String fullyQualifiedName = m.group(1);
      return getInferredType(fullyQualifiedName);
    }
    return null;
  }

  private static List<String> parseTopLevelTypeString(String typeList) {
    List<String> innerTypeNames = new ArrayList<>();
    int parentheses = 0;
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < typeList.length(); i++) {
      char c = typeList.charAt(i);
      switch (c) {
        case ',':
          if (parentheses == 0) {
            innerTypeNames.add(sb.toString());
            sb.setLength(0);
            break;
          }
          sb.append(c);
          break;
        case '(':
          parentheses += 1;
          sb.append(c);
          break;
        case ')':
          parentheses -= 1;
          sb.append(c);
          break;
        case ' ':
          break;
        default:
          sb.append(c);
      }

    }
    if (parentheses != 0) {
      throw new RecognitionException("Unbalanced parentheses", null, null, null);
    }
    if (sb.length() > 0) {
      innerTypeNames.add(sb.toString());
    }
    return innerTypeNames;
  }

  private static InferredType getInferredType(String fullyQualifiedName) {
    return Optional.ofNullable(getRuntimeType(fullyQualifiedName)).orElseGet(() -> {
      //      LOG.error("");
      //      LOG.error(String.format("Unresolved class symbol: %s", fullyQualifiedName));
      //      LOG.error(String.format("Filename: %s", fileName));
      //      LOG.error("");
      return InferredTypes.anyType();
    });
  }

  private static InferredType getRuntimeType(String fullyQualifiedName) {
    if (fullyQualifiedName.equals("typing.Callable")) {
      return new RuntimeType(org.sonar.python.types.TypeContext.CALLABLE_CLASS_SYMBOL);
    }
    Symbol symbol = projectLevelSymbolTable.getSymbol(fullyQualifiedName);
    if (symbol != null && symbol.is(Symbol.Kind.CLASS) && symbol.fullyQualifiedName() != null) {
      return new RuntimeType(((ClassSymbol) symbol));
    }
    symbol = TypeShed.symbolWithFQN(fullyQualifiedName);
    if (symbol != null && symbol.is(Symbol.Kind.CLASS) && symbol.fullyQualifiedName() != null) {
      return new RuntimeType(((ClassSymbol) symbol));
    }
    return null;
  }

}
