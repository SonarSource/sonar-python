/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.TokenLocation;
import org.sonar.python.api.PythonTokenType;

public class TreeUtils {
  private TreeUtils() {
    // empty constructor
  }

  private static final Set<PythonTokenType> WHITESPACE_TOKEN_TYPES = EnumSet.of(
    PythonTokenType.NEWLINE,
    PythonTokenType.INDENT,
    PythonTokenType.DEDENT);

  @CheckForNull
  public static Tree firstAncestor(Tree tree, Predicate<Tree> predicate) {
    Tree currentParent = tree.parent();
    while (currentParent != null) {
      if (predicate.test(currentParent)) {
        return currentParent;
      }
      currentParent = currentParent.parent();
    }
    return null;
  }

  @CheckForNull
  public static Tree firstAncestorOfKind(Tree tree, Kind... kinds) {
    return firstAncestor(tree, t -> t.is(kinds));
  }

  public static List<Token> tokens(Tree tree) {
    if (tree.is(Kind.TOKEN)) {
      return Collections.singletonList((Token) tree);
    } else if (tree.is(Kind.STRING_ELEMENT)) {
      return Collections.singletonList(tree.firstToken());
    }
    List<Token> tokens = new ArrayList<>();
    for (Tree child : tree.children()) {
      if (child.is(Kind.TOKEN)) {
        tokens.add(((Token) child));
      } else {
        tokens.addAll(tokens(child));
      }
    }
    return tokens;
  }

  public static List<Token> nonWhitespaceTokens(Tree tree) {
    return TreeUtils.tokens(tree).stream()
      .filter(t -> !WHITESPACE_TOKEN_TYPES.contains(t.type()))
      .collect(Collectors.toList());
  }

  public static boolean hasDescendant(Tree tree, Predicate<Tree> predicate) {
    return tree.children().stream().anyMatch(child -> predicate.test(child) || hasDescendant(child, predicate));
  }

  public static Stream<Expression> flattenTuples(Expression expression) {
    if (expression.is(Kind.TUPLE)) {
      Tuple tuple = (Tuple) expression;
      return tuple.elements().stream().flatMap(TreeUtils::flattenTuples);
    } else {
      return Stream.of(expression);
    }
  }

  public static Optional<Symbol> getSymbolFromTree(@Nullable Tree tree) {
    if (tree instanceof HasSymbol) {
      return Optional.ofNullable(((HasSymbol) tree).symbol());
    }
    return Optional.empty();
  }

  @CheckForNull
  public static ClassSymbol getClassSymbolFromDef(@Nullable ClassDef classDef) {
    if (classDef == null) {
      return null;
    }

    Symbol classNameSymbol = classDef.name().symbol();
    if (classNameSymbol == null) {
      throw new IllegalStateException("A ClassDef should always have a non-null symbol!");
    }

    if (classNameSymbol.kind() == Symbol.Kind.CLASS) {
      return ((ClassSymbol) classNameSymbol);
    }

    return null;
  }

  @CheckForNull
  public static FunctionSymbol getFunctionSymbolFromDef(@Nullable FunctionDef functionDef) {
    if (functionDef == null) {
      return null;
    }

    Symbol functionNameSymbol = functionDef.name().symbol();
    if (functionNameSymbol == null) {
      throw new IllegalStateException("A FunctionDef should always have a non-null symbol!");
    }

    if (functionNameSymbol.kind() == Symbol.Kind.FUNCTION) {
      return ((FunctionSymbol) functionNameSymbol);
    }

    return null;
  }

  public static List<Parameter> nonTupleParameters(FunctionDef functionDef) {
    ParameterList parameterList = functionDef.parameters();
    if (parameterList == null) {
      return Collections.emptyList();
    }
    return parameterList.nonTuple();
  }

  public static List<Parameter> positionalParameters(FunctionDef functionDef) {
    ParameterList parameterList = functionDef.parameters();
    if (parameterList == null) {
      return Collections.emptyList();
    }

    List<Parameter> result = new ArrayList<>();
    for (AnyParameter anyParameter : parameterList.all()) {
      if (anyParameter instanceof Parameter) {
        Parameter parameter = (Parameter) anyParameter;
        Token starToken = parameter.starToken();
        if (parameter.name() == null && starToken != null) {
          if ("*".equals(starToken.value())) {
            return result;
          }
          // Ignore the possible '/' parameter
        } else {
          result.add(parameter);
        }
      }
    }

    return result;
  }

  /**
   * Collects all top-level function definitions within a class def.
   * It is used to discover methods defined within "strange" constructs, such as
   * <code>
   *   class A:
   *       if p:
   *           def f(self): ...
   * </code>
   */
  public static List<FunctionDef> topLevelFunctionDefs(ClassDef classDef) {
    CollectFunctionDefsVisitor visitor = new CollectFunctionDefsVisitor();
    classDef.body().accept(visitor);

    return visitor.functionDefs;
  }

  private static class CollectFunctionDefsVisitor extends BaseTreeVisitor {
    private List<FunctionDef> functionDefs = new ArrayList<>();

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      // Do not descend into nested classes
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      this.functionDefs.add(pyFunctionDefTree);
      // Do not descend into nested functions
    }
  }

  @CheckForNull
  public static RegularArgument argumentByKeyword(String keyword, List<Argument> arguments) {
    for (int i = 0; i < arguments.size(); i++) {
      Argument argument = arguments.get(i);
      if (hasKeyword(argument, keyword)) {
        return ((RegularArgument) argument);
      }
    }
    return null;
  }

  @CheckForNull
  public static RegularArgument nthArgumentOrKeyword(int argPosition, String keyword, List<Argument> arguments) {
    for (int i = 0; i < arguments.size(); i++) {
      Argument argument = arguments.get(i);
      if (hasKeyword(argument, keyword)) {
        return ((RegularArgument) argument);
      }
      if (argument.is(Kind.REGULAR_ARGUMENT)) {
        RegularArgument regularArgument = (RegularArgument) argument;
        if (regularArgument.keywordArgument() == null && argPosition == i) {
          return regularArgument;
        }
      }
    }
    return null;
  }

  private static boolean hasKeyword(Argument argument, String keyword) {
    if (argument.is(Kind.REGULAR_ARGUMENT)) {
      Name keywordArgument = ((RegularArgument) argument).keywordArgument();
      return keywordArgument != null && keywordArgument.name().equals(keyword);
    }
    return false;
  }

  public static boolean isBooleanLiteral(Tree tree) {
    if (tree.is(Kind.NAME)) {
      String name = ((Name) tree).name();
      return name.equals("True") || name.equals("False");
    }
    return false;
  }

  public static String nameFromExpression(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return ((Name) expression).name();
    }
    return null;
  }

  private static String nameFromQualifiedExpression(QualifiedExpression qualifiedExpression) {
    String exprName = qualifiedExpression.name().name();
    Expression qualifier = qualifiedExpression.qualifier();
    String nameOfQualifier = decoratorNameFromExpression(qualifier);
    if (nameOfQualifier != null) {
      exprName = nameOfQualifier + "." + exprName;
    } else {
      exprName = null;
    }
    return exprName;
  }

  @CheckForNull
  public static String decoratorNameFromExpression(Expression expression) {
    if (expression.is(Kind.NAME)) {
      return ((Name) expression).name();
    }
    if (expression.is(Kind.QUALIFIED_EXPR)) {
      return nameFromQualifiedExpression((QualifiedExpression) expression);
    }
    if (expression.is(Kind.CALL_EXPR)) {
      return decoratorNameFromExpression(((CallExpression) expression).callee());
    }
    return null;
  }

  @CheckForNull
  public static LocationInFile locationInFile(Tree tree, @Nullable String fileId) {
    if (fileId == null) {
      return null;
    }
    TokenLocation firstToken = new TokenLocation(tree.firstToken());
    TokenLocation lastToken = new TokenLocation(tree.lastToken());
    return new LocationInFile(fileId, firstToken.startLine(), firstToken.startLineOffset(), lastToken.endLine(), lastToken.endLineOffset());
  }
}
