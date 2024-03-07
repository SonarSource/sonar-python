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

import java.util.Objects;
import java.util.Optional;
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

  public static InferredType getTypeFromString(String detailedType) {
    TypeContext typeContext = getParseTree(detailedType);
    return getTypeFromParseTree(typeContext);
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
