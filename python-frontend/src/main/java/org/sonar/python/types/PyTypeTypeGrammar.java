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

import com.google.common.eventbus.DeadEvent;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.RecognitionException;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.pytype_grammar.ExceptionErrorStrategy;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarLexer;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Builtin_typeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Class_typeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Generic_typeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Qualified_typeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.TypeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Union_typeContext;

public class PyTypeTypeGrammar {

  static ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
  static String fileName = "";

  private static final Logger LOG = LoggerFactory.getLogger(PyTypeTypeGrammar.class);

  public static TypeContext getParseTree(String typeString) throws RecognitionException {
    PyTypeTypeGrammarLexer lexer = new PyTypeTypeGrammarLexer(CharStreams.fromString(typeString));
    PyTypeTypeGrammarParser parser = new PyTypeTypeGrammarParser(new CommonTokenStream(lexer));
    parser.setErrorHandler(new ExceptionErrorStrategy());
    return parser.outer_type().type();
  }


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
        var parameters = parseTopLevelTypeString(m.group(2));
        return getTypeFromString(parameters.get(0));
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

  public static InferredType getTypeFromParseTree(TypeContext typeContext) {
    if (typeContext.qualified_type() != null) {
      return getInferredTypeForQualified(typeContext.qualified_type());
    } else if (typeContext.builtin_type() != null) {
      return getInferredTypeForBuiltin(typeContext.builtin_type());
    } else if (typeContext.class_type() != null) {
      return getInferredTypeForClass(typeContext.class_type());
    } else if (typeContext.generic_type() != null) {
      return getInferredTypeForGeneric(typeContext.generic_type());
    } else if (typeContext.union_type() != null) {
      return getInferredTypeForUnion(typeContext);
    } else if (typeContext.anything_type() != null) {
      return InferredTypes.anyType();
    } else {
      return null;
    }
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

  private static InferredType getInferredTypeForGeneric(Generic_typeContext genericTypeContext) {
    TypeContext type = genericTypeContext.type();
    if (type != null) {
      return getTypeFromParseTree(type);
    }
    return null;
  }

  private static InferredType getInferredTypeForQualified(Qualified_typeContext qualifiedTypeContext) {
    String fullyQualifiedName = qualifiedTypeContext.STRING().stream().map(TerminalNode::toString).collect(Collectors.joining("."));
    return getInferredType(fullyQualifiedName);
  }

  private static InferredType getInferredTypeForUnion(TypeContext typeContext) {
    Union_typeContext unionTypeContext = typeContext.union_type();
    Stream<InferredType> unionTypeSet = unionTypeContext.type_list().type().stream().map(PyTypeTypeGrammar::getTypeFromParseTree).filter(Objects::nonNull);
    return InferredTypes.union(unionTypeSet);
  }

  private static InferredType getInferredTypeForClass(Class_typeContext classTypeContext) {
    if (classTypeContext.builtin_type() != null) {
      return getInferredTypeForBuiltin(classTypeContext.builtin_type());
    } else {
      Qualified_typeContext qualifiedTypeContext = classTypeContext.qualified_type();
      return getInferredTypeForQualified(qualifiedTypeContext);
    }
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

  private static InferredType getInferredTypeForBuiltin(Builtin_typeContext builtinTypeContext) {
    return Stream.of(
      builtinTypeContext.builtin().NONE_TYPE(),
      builtinTypeContext.builtin().BOOL(),
      builtinTypeContext.builtin().STR(),
      builtinTypeContext.builtin().INT(),
      builtinTypeContext.builtin().FLOAT(),
      builtinTypeContext.builtin().COMPLEX(),
      builtinTypeContext.builtin().TUPLE(),
      builtinTypeContext.builtin().LIST(),
      builtinTypeContext.builtin().SET(),
      builtinTypeContext.builtin().DICT(),
      builtinTypeContext.builtin().BASE_EXCEPTION()
    ).filter(Objects::nonNull)
      .map(Objects::toString)
      .map(InferredTypes::runtimeBuiltinType)
      .findFirst()
      .orElse(null);
  }

}
