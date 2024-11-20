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
package org.sonar.python.checks.utils;

import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.GENERATOR_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.LAMBDA;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NONE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NUMERIC_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STRING_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.UNPACKING_EXPR;

public class CheckUtils {

  private CheckUtils() {

  }

  public static boolean areEquivalent(@Nullable Tree leftTree, @Nullable Tree rightTree) {
    if (leftTree == rightTree) {
      return true;
    }
    if (leftTree == null || rightTree == null) {
      return false;
    }
    if (leftTree.getKind() != rightTree.getKind() || leftTree.children().size() != rightTree.children().size()) {
      return false;
    }
    if (leftTree.children().isEmpty() && rightTree.children().isEmpty()) {
      return areLeavesEquivalent(leftTree, rightTree);
    }

    List<Tree> children1 = leftTree.children();
    List<Tree> children2 = rightTree.children();
    for (int i = 0; i < children1.size(); i++) {
      if (!areEquivalent(children1.get(i), children2.get(i))) {
        return false;
      }
    }
    return true;
  }

  private static boolean areLeavesEquivalent(Tree leftLeaf, Tree rightLeaf) {
    if (leftLeaf.firstToken() == null && rightLeaf.firstToken() == null) {
      return true;
    }
    return leftLeaf.firstToken().type().equals(PythonTokenType.INDENT) || leftLeaf.firstToken().type().equals(PythonTokenType.DEDENT) ||
      leftLeaf.firstToken().value().equals(rightLeaf.firstToken().value());
  }

  @CheckForNull
  public static ClassDef getParentClassDef(Tree tree) {
    Tree current = tree.parent();
    while (current != null) {
      if (current.is(Tree.Kind.CLASSDEF)) {
        return (ClassDef) current;
      } else if (current.is(Tree.Kind.FUNCDEF, Tree.Kind.LAMBDA)) {
        return null;
      }
      current = current.parent();
    }
    return null;
  }

  public static boolean classHasInheritance(ClassDef classDef) {
    ArgList argList = classDef.args();
    if (argList == null) {
      return false;
    }
    List<Argument> arguments = argList.arguments();
    if (arguments.isEmpty()) {
      return false;
    }
    return arguments.size() != 1 || !"object".equals(arguments.get(0).firstToken().value());
  }

  public static boolean containsCallToLocalsFunction(Tree tree) {
    return TreeUtils.hasDescendant(tree, t -> t.is(Tree.Kind.CALL_EXPR) && calleeHasNameLocals(((CallExpression) t)));
  }

  private static boolean calleeHasNameLocals(CallExpression callExpression) {
    Expression callee = callExpression.callee();
    return callee.is(Tree.Kind.NAME) && "locals".equals(((Name) callee).name());
  }

  public static boolean isConstant(Expression condition) {
    return isImmutableConstant(condition) || isConstantCollectionLiteral(condition);
  }

  public static boolean isImmutableConstant(Expression condition) {
    return TreeUtils.isBooleanLiteral(condition) ||
      condition.is(NUMERIC_LITERAL, STRING_LITERAL, NONE, LAMBDA, GENERATOR_EXPR);
  }

  public static boolean isConstantCollectionLiteral(Expression condition) {
    switch (condition.getKind()) {
      case LIST_LITERAL:
        return doesNotContainUnpackingExpression(((ListLiteral) condition).elements().expressions());
      case DICTIONARY_LITERAL:
        return doesNotContainUnpackingExpression(((DictionaryLiteral) condition).elements());
      case SET_LITERAL:
        return doesNotContainUnpackingExpression(((SetLiteral) condition).elements());
      case TUPLE:
        return doesNotContainUnpackingExpression(((Tuple) condition).elements());
      default:
        return false;
    }
  }

  private static boolean doesNotContainUnpackingExpression(List<? extends Tree> elements) {
    if (elements.isEmpty()) {
      return true;
    }
    return elements.stream().anyMatch(element -> !element.is(UNPACKING_EXPR));
  }

  public static boolean isNone(InferredType type) {
    return type.canOnlyBe(BuiltinTypes.NONE_TYPE);
  }

  private static final List<String> PROTOCOL_LIKE_BASE_TYPES = List.of("typing.Protocol", "zope.interface.Interface");

  /**
   * Determines whether the given class must be a <a href="https://docs.python.org/3/library/typing.html#typing.Protocol">Protocol</a>
   * or a similar Protocol-like definition (e.g. {@code zope.interface.Interface}).
   */
  public static boolean mustBeAProtocolLike(ClassDef classDef) {
    var classSymbol = TreeUtils.getClassSymbolFromDef(classDef);
    if (classSymbol != null) {
      return PROTOCOL_LIKE_BASE_TYPES.stream().anyMatch(classSymbol::isOrExtends);
    }

    return false;
  }

  private static final List<String> ABC_ABSTRACTMETHOD_DECORATORS = List.of("abstractmethod", "abc.abstractmethod");

  public static boolean isAbstract(FunctionDef funDef) {
    return funDef
      .decorators()
      .stream()
      .map(decorator -> TreeUtils.decoratorNameFromExpression(decorator.expression()))
      .anyMatch(foundDeco -> ABC_ABSTRACTMETHOD_DECORATORS.stream().anyMatch(abcDeco -> abcDeco.equals(foundDeco)));
  }

  /**
   * Simple check whether the given expression is the "self" name expression.
   *
   * Carefully check the context when relying on this method!
   * This implementation does not ensure that the name is actually referring to a method parameter or whether the surrounding method might
   * be static, etc.
   */
  public static boolean isSelf(Expression expression) {
    // TODO: Instead of performing a manual string comparison, maybe check the symbol instead for being a SelfSymbolImpl symbol
    // (This might require exposing some more information about the symbol kind being a "Self"-symbol)
    return expression.is(NAME) && "self".equals(((Name) expression).name());
  }

  @CheckForNull
  public static Symbol findFirstParameterSymbol(FunctionDef functionDef) {
    ParameterList parameters = functionDef.parameters();
    if (parameters == null) {
      return null;
    }
    List<Parameter> params = parameters.nonTuple();
    if (params.isEmpty()) {
      return null;
    }
    Name firstParameterName = params.get(0).name();
    if (firstParameterName == null) {
      return null;
    }

    return firstParameterName.symbol();
  }
}
