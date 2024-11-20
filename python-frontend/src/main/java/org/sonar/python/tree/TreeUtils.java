/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collector;
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
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
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

  public static Collector<Tree, ?, Map<Tree, Tree>> groupAssignmentByParentStatementList() {
    return Collectors.toMap(tree -> TreeUtils.firstAncestor(tree, parent -> parent.is(Tree.Kind.STATEMENT_LIST)),
      Function.identity(),
      //Get just first element for each block
      (t1, t2) ->
        Stream.of(t1, t2).min(getTreeByPositionComparator()).get());
  }

  public static Comparator<Tree> getTreeByPositionComparator() {
    return Comparator.comparing((Tree t) -> t.firstToken().line()).thenComparing((Tree t) -> t.firstToken().column());
  }

  public static List<Token> tokens(Tree tree) {
    if (tree.is(Kind.TOKEN)) {
      return Collections.singletonList((Token) tree);
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
      .toList();
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
    if (tree instanceof HasSymbol hasSymbol) {
      return Optional.ofNullable(hasSymbol.symbol());
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

  public static List<String> getParentClassesFQN(ClassDef classDef) {
    return getParentClasses(TreeUtils.getClassSymbolFromDef(classDef), new HashSet<>()).stream()
      .map(Symbol::fullyQualifiedName)
      .filter(Objects::nonNull)
      .toList();
  }

  private static List<Symbol> getParentClasses(@Nullable ClassSymbol classSymbol, Set<ClassSymbol> visitedSymbols) {
    List<Symbol> superClasses = new ArrayList<>();
    if (classSymbol == null || visitedSymbols.contains(classSymbol)) {
      return superClasses;
    }
    visitedSymbols.add(classSymbol);
    for (Symbol symbol : classSymbol.superClasses()) {
      superClasses.add(symbol);
      if (symbol instanceof ClassSymbol superClassSymbol) {
        superClasses.addAll(getParentClasses(superClassSymbol, visitedSymbols));
      }
    }
    return superClasses;
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
      if (anyParameter instanceof Parameter parameter) {
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

  public static int findIndentationSize(Tree tree) {
    var parent = tree.parent();

    if (parent == null) {
      return findIndentDownTree(tree);
    }

    var treeToken = tree.firstToken();
    var parentToken = parent.firstToken();

    if (treeToken.line() != parentToken.line()) {
      return treeToken.column() - parentToken.column();
    } else {
      return findIndentationSize(parent);
    }
  }

  private static int findIndentDownTree(Tree parent) {
    var parentToken = parent.firstToken();
    return parent.children()
      .stream()
      .map(child -> {

        var childToken = child.firstToken();
        if (childToken.line() > parentToken.line() && childToken.column() > parentToken.column()) {
          return childToken.column() - parentToken.column();
        } else  {
          return findIndentDownTree(child);
        }
      })
      .filter(i -> i > 0)
      .findFirst()
      .orElse(0);
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

  public static Optional<String> nameFromQualifiedOrCallExpression(Expression expression) {
    return Optional.ofNullable(TreeUtils.nameFromExpression(expression))
      .or(() -> TreeUtils.toOptionalInstanceOf(QualifiedExpression.class, expression)
        .map(TreeUtils::nameFromQualifiedExpression))
      .or(() -> TreeUtils.toOptionalInstanceOf(CallExpression.class, expression)
        .map(CallExpression::callee)
        .flatMap(TreeUtils::nameFromExpressionOrQualifiedExpression));
  }

  public static Optional<String> nameFromExpressionOrQualifiedExpression(Expression expression) {
    return TreeUtils.toOptionalInstanceOf(QualifiedExpression.class, expression)
      .map(TreeUtils::nameFromQualifiedExpression)
      .or(() -> Optional.ofNullable(TreeUtils.nameFromExpression(expression)));
  }

  public static String nameFromQualifiedExpression(QualifiedExpression qualifiedExpression) {
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


  public static boolean isFunctionWithGivenDecoratorFQN(Tree tree, String decoratorFQN) {
    if (!tree.is(Kind.FUNCDEF)) {
      return false;
    }
    return ((FunctionDef) tree).decorators().stream().anyMatch(d -> isDecoratorWithFQN(d, decoratorFQN));
  }

  public static boolean isDecoratorWithFQN(Decorator decorator, String fullyQualifiedName) {
    return Optional.of(decorator.expression())
      .flatMap(TreeUtils::getSymbolFromTree)
      .map(Symbol::fullyQualifiedName)
      .filter(fullyQualifiedName::equals)
      .isPresent();
  }

  public static Optional<String> fullyQualifiedNameFromQualifiedExpression(QualifiedExpression qualifiedExpression) {
    String exprName = qualifiedExpression.name().name();
    Expression qualifier = qualifiedExpression.qualifier();
    return fullyQualifiedNameFromExpression(qualifier).map(nameOfQualifier -> nameOfQualifier + "." + exprName);
  }

  public static Optional<String> fullyQualifiedNameFromExpression(Expression expression) {
    if (expression.is(Kind.NAME)) {
      Symbol symbol = ((Name) expression).symbol();
      return Optional.of(Optional.ofNullable(symbol).map(Symbol::fullyQualifiedName).orElse(((Name) expression).name()));
    }
    if (expression.is(Kind.QUALIFIED_EXPR)) {
      return fullyQualifiedNameFromQualifiedExpression((QualifiedExpression) expression);
    }
    if (expression.is(Kind.CALL_EXPR)) {
      return fullyQualifiedNameFromExpression(((CallExpression) expression).callee());
    }
    return Optional.empty();
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

  /**
   * Statements can have a separator like semicolon. When handling ranges we want to take them into account.
   */
  public static Token getTreeSeparatorOrLastToken(Tree tree) {
    if (tree instanceof Statement statement) {
      Token separator = statement.separator();
      if (separator != null) {
        return separator;
      }
    }
    return tree.lastToken();
  }

  public static <T extends Tree> Function<Tree, T> toInstanceOfMapper(Class<T> castToClass) {
    return toOptionalInstanceOfMapper(castToClass).andThen(t -> t.orElse(null));
  }

  public static <T extends Tree> Function<Tree, Optional<T>> toOptionalInstanceOfMapper(Class<T> castToClass) {
    return tree -> toOptionalInstanceOf(castToClass, tree);
  }

  public static <T extends Tree> Optional<T> toOptionalInstanceOf(Class<T> castToClass, @Nullable Tree tree) {
    return Optional.ofNullable(tree).filter(castToClass::isInstance).map(castToClass::cast);
  }

  public static <T extends Tree> Function<Tree, Stream<T>> toStreamInstanceOfMapper(Class<T> castToClass) {
    return tree -> toOptionalInstanceOf(castToClass, tree).map(Stream::of).orElse(Stream.empty());
  }

  public static Optional<Tree> firstChild(Tree tree, Predicate<Tree> filter) {
    if (filter.test(tree)) {
      return Optional.of(tree);
    }
    return tree.children()
      .stream()
      .map(c -> firstChild(c, filter))
      .filter(Optional::isPresent)
      .findFirst()
      .map(Optional::get);
  }

  public static String treeToString(Tree tree, boolean renderMultiline) {
    if (!renderMultiline) {
      var firstLine = tree.firstToken().line();
      var lastLine = tree.lastToken().line();

      // We decided to not support multiline default parameters
      // because it requires indents calculation for place where the value should be copied.
      if (firstLine != lastLine) {
        return null;
      }
    }

    var tokens = TreeUtils.tokens(tree);

    var valueBuilder = new StringBuilder();
    for (int i = 0; i < tokens.size(); i++) {
      var token = tokens.get(i);
      if (i > 0) {
        var previous = tokens.get(i - 1);
        var spaceBetween = token.column() - previous.column() - previous.value().length();
        if (spaceBetween < 0) {
          spaceBetween = token.column();
        }
        valueBuilder.append(" ".repeat(spaceBetween));
      }
      valueBuilder.append(token.value());
    }
    return valueBuilder.toString();
  }


  public static List<String> dottedNameToPartFqn(DottedName dottedName) {
    return dottedName.names()
      .stream()
      .map(Name::name)
      .toList();
  }

}
