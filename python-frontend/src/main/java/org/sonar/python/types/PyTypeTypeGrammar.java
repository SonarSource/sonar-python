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

import java.util.List;
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
    List<String> collected = qualifiedTypeContext.STRING().stream().map(TerminalNode::toString).collect(Collectors.toList());
    ClassSymbolImpl classSymbol = new ClassSymbolImpl(collected.get(collected.size() - 1), String.join(".", collected));
    return new RuntimeType(classSymbol);
  }

  private static InferredType getInferredTypeForUnion(TypeContext typeContext) {
    Union_typeContext unionTypeContext = typeContext.union_type();
    Stream<InferredType> unionTypeSet = unionTypeContext.type_list().type().stream().map(PyTypeTypeGrammar::getTypeFromParseTree);
    return InferredTypes.union(unionTypeSet);
  }

  private static InferredType getInferredTypeForClass(Class_typeContext classTypeContext) {
    if (classTypeContext.builtin_type() != null) {
      return getInferredTypeForBuiltin(classTypeContext.builtin_type());
    } else {
      Qualified_typeContext qualifiedTypeContext = classTypeContext.qualified_type();
      String fullyQualifiedName = qualifiedTypeContext.STRING().stream().map(TerminalNode::toString).collect(Collectors.joining("."));
      if (fullyQualifiedName.equals("typing.Callable")) {
        return new RuntimeType(org.sonar.python.types.TypeContext.CALLABLE_CLASS_SYMBOL);
      }
      Symbol symbol = projectLevelSymbolTable.getSymbol(fullyQualifiedName);
      if (symbol != null && symbol.is(Symbol.Kind.CLASS)) {
        return new RuntimeType(((ClassSymbol) symbol));
      }
      symbol = TypeShed.symbolWithFQN(fullyQualifiedName);
      if (symbol != null && symbol.is(Symbol.Kind.CLASS)) {
        return new RuntimeType(((ClassSymbol) symbol));
      }
      LOG.error("");
      LOG.error(String.format("Unresolved class symbol: %s", fullyQualifiedName));
      LOG.error(String.format("Filename: %s", fileName));
      LOG.error("");
      return InferredTypes.anyType();
    }
  }

  private static InferredType getInferredTypeForBuiltin(Builtin_typeContext builtinTypeContext) {
    if (builtinTypeContext.builtin().NONE_TYPE() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().NONE_TYPE().toString());
    } else if (builtinTypeContext.builtin().BOOL() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().BOOL().toString());
    } else if (builtinTypeContext.builtin().STR() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().STR().toString());
    } else if (builtinTypeContext.builtin().INT() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().INT().toString());
    } else if (builtinTypeContext.builtin().FLOAT() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().FLOAT().toString());
    } else if (builtinTypeContext.builtin().COMPLEX() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().COMPLEX().toString());
    } else if (builtinTypeContext.builtin().TUPLE() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().TUPLE().toString());
    } else if (builtinTypeContext.builtin().LIST() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().LIST().toString());
    } else if (builtinTypeContext.builtin().SET() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().SET().toString());
    } else if (builtinTypeContext.builtin().DICT() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().DICT().toString());
    } else if (builtinTypeContext.builtin().BASE_EXCEPTION() != null) {
      return InferredTypes.runtimeBuiltinType(builtinTypeContext.builtin().BASE_EXCEPTION().toString());
    } else {
      return null;
    }
  }

}
