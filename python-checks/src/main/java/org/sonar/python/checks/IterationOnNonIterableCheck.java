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
package org.sonar.python.checks;

import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S3862")
public class IterationOnNonIterableCheck extends IterationOnNonIterable {

  private static final String MESSAGE = "Replace this expression with an iterable object.";

  boolean isValidIterable(Expression expression, Map<LocationInFile, String> secondaries) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      CallExpression callExpression = (CallExpression) expression;
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && calleeSymbol.is(Symbol.Kind.FUNCTION) && ((FunctionSymbol) calleeSymbol).isAsynchronous()) {
        FunctionSymbol functionSymbol = (FunctionSymbol) calleeSymbol;
        secondaries.put(functionSymbol.definitionLocation(), String.format(SECONDARY_MESSAGE, functionSymbol.name()));
        return false;
      }
    }
    if (expression instanceof HasSymbol hasSymbol) {
      Symbol symbol = hasSymbol.symbol();
      if (symbol != null) {
        if (symbol.is(Symbol.Kind.FUNCTION)) {
          FunctionSymbol functionSymbol = (FunctionSymbol) symbol;
          secondaries.put(functionSymbol.definitionLocation(), String.format(SECONDARY_MESSAGE, functionSymbol.name()));
          return functionSymbol.hasDecorators();
        }
        if (symbol.is(Symbol.Kind.CLASS)) {
          ClassSymbolImpl classSymbol = (ClassSymbolImpl) symbol;
          secondaries.put(classSymbol.definitionLocation(), String.format(SECONDARY_MESSAGE, classSymbol.name()));
          // Metaclasses might add the method by default
          return classSymbol.hasSuperClassWithUnknownMetaClass() || classSymbol.hasUnresolvedTypeHierarchy();
        }
      }
    }
    InferredType type = expression.type();
    String typeName = InferredTypes.typeName(type);
    String secondaryMessage = typeName != null ? String.format(SECONDARY_MESSAGE, typeName) : DEFAULT_SECONDARY_MESSAGE;
    secondaries.put(InferredTypes.typeClassLocation(type), secondaryMessage);
    return type.canHaveMember("__iter__") || type.canHaveMember("__getitem__");
  }

  @Override
  boolean isAsyncIterable(Expression expression) {
    if (expression.is(Tree.Kind.CALL_EXPR)) {
      CallExpression callExpression = (CallExpression) expression;
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && calleeSymbol.is(Symbol.Kind.FUNCTION)) {
        return ((FunctionSymbol) calleeSymbol).isAsynchronous();
      }
    }
    return expression.type().canHaveMember("__aiter__");
  }

  @Override
  String message(Expression expression, boolean isForLoop) {
    return isForLoop && isAsyncIterable(expression) ? "Add \"async\" before \"for\"; This expression is an async generator." : MESSAGE;
  }
}
