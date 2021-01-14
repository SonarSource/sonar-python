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
package org.sonar.python.checks;

import java.util.Optional;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.types.InferredTypes.typeName;

@Rule(key = "S5655")
public class ArgumentTypeCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol == null) {
        return;
      }
      if (!calleeSymbol.is(Symbol.Kind.FUNCTION)) {
        // We might want to support ambiguous symbols for which every definition is a function
        return;
      }
      FunctionSymbol functionSymbol = (FunctionSymbol) calleeSymbol;
      if (functionSymbol.hasVariadicParameter()) {
        return;
      }
      boolean isStaticCall = callExpression.callee().is(Tree.Kind.NAME) || Optional.of(callExpression.callee())
        .filter(c -> c.is(Tree.Kind.QUALIFIED_EXPR))
        .flatMap(q -> TreeUtils.getSymbolFromTree(((QualifiedExpression) q).qualifier()).filter(s -> s.is(Symbol.Kind.CLASS)))
        .isPresent();
      checkFunctionCall(ctx, callExpression, functionSymbol, isStaticCall);
    });
  }

  private static void checkFunctionCall(SubscriptionContext ctx, CallExpression callExpression, FunctionSymbol functionSymbol, boolean isStaticCall) {

    boolean isKeyword = false;
    int firstParameterOffset = SymbolUtils.firstParameterOffset(functionSymbol, isStaticCall);
    if (firstParameterOffset < 0) {
      return;
    }
    for (int i = 0; i < callExpression.arguments().size(); i++) {
      Argument argument = callExpression.arguments().get(i);
      int parameterIndex = i + firstParameterOffset;
      if (parameterIndex >= functionSymbol.parameters().size()) {
        // S930 will raise the issue
        return;
      }
      if (argument.is(Tree.Kind.REGULAR_ARGUMENT)) {
        RegularArgument regularArgument = (RegularArgument) argument;
        isKeyword |= regularArgument.keywordArgument() != null;
        boolean shouldReport = isKeyword ? shouldReportKeywordArgument(regularArgument, functionSymbol)
          : shouldReportPositionalArgument(regularArgument, functionSymbol, parameterIndex);
        if (shouldReport) {
          reportIssue(ctx, functionSymbol, regularArgument);
        }
      }
    }
  }

  private static boolean shouldReportPositionalArgument(RegularArgument regularArgument, FunctionSymbol functionSymbol, int index) {
    FunctionSymbol.Parameter functionParameter = functionSymbol.parameters().get(index);
    InferredType argumentType = regularArgument.expression().type();
    InferredType parameterType = functionParameter.declaredType();
    if (parameterType.canOnlyBe("object")) {
      // Avoid FPs as every Python 3 class implicitly inherits from object
      return false;
    }
    return isIncompatibleTypes(argumentType, parameterType);
  }

  private static boolean shouldReportKeywordArgument(RegularArgument regularArgument, FunctionSymbol functionSymbol) {
    Name keywordArgument = regularArgument.keywordArgument();
    InferredType argumentType = regularArgument.expression().type();
    if (keywordArgument == null) {
      // Syntax error
      return false;
    }
    String keywordName = keywordArgument.name();
    Optional<FunctionSymbol.Parameter> correspondingParameter = functionSymbol.parameters().stream().filter(p -> keywordName.equals(p.name())).findFirst();
    return correspondingParameter
      .map(c -> {
        InferredType parameterType = c.declaredType();
        return (isIncompatibleTypes(argumentType, parameterType));
      })
      // If not present: S930 will raise the issue
      .orElse(false);
  }

  private static void reportIssue(SubscriptionContext ctx, FunctionSymbol functionSymbol, RegularArgument regularArgument) {
    PreciseIssue issue = ctx.addIssue(regularArgument, String.format("Change this argument; Function \"%s\" expects a different type", functionSymbol.name()));
    LocationInFile locationInFile = functionSymbol.definitionLocation();
    if (locationInFile != null) {
      issue.secondary(locationInFile, "Function definition");
    }
  }

  private static boolean isIncompatibleTypes(InferredType argumentType, InferredType parameterType) {
    return (isNotDuckTypeCompatible(argumentType, parameterType)
      || (!argumentType.isCompatibleWith(parameterType) && !couldBeDuckTypeCompatible(argumentType, parameterType))) && !isException(argumentType);
  }

  private static boolean isNotDuckTypeCompatible(InferredType argumentType, InferredType parameterType) {
    // Avoid FNs if builtins have incomplete type hierarchy when we are certain of their type
    String firstBuiltin = matchBuiltinCategory(name -> name.equals(typeName(argumentType)));
    String secondBuiltin = matchBuiltinCategory(name -> name.equals(typeName(parameterType)));
    return firstBuiltin != null && secondBuiltin != null && !firstBuiltin.equals(secondBuiltin);
  }

  private static boolean couldBeDuckTypeCompatible(InferredType firstType, InferredType secondType) {
    // Here we'll return true if we cannot exclude possible duck typing because of unresolved type hierarchies or typing aliases
    String firstPossibleBuiltin = matchBuiltinCategory(firstType::canBeOrExtend);
    String secondPossibleBuiltin = matchBuiltinCategory(secondType::canBeOrExtend);
    return firstPossibleBuiltin != null && firstPossibleBuiltin.equals(secondPossibleBuiltin);
  }

  public static String matchBuiltinCategory(Predicate<String> predicate) {
    if (predicate.test(BuiltinTypes.STR)) {
      return BuiltinTypes.STR;
    }
    if (predicate.test(BuiltinTypes.INT)
      || predicate.test(BuiltinTypes.FLOAT)
      || predicate.test(BuiltinTypes.COMPLEX)
      || predicate.test(BuiltinTypes.BOOL)) {
      return "number";
    }
    if (predicate.test(BuiltinTypes.LIST)) {
      return BuiltinTypes.LIST;
    }
    if (predicate.test(BuiltinTypes.SET)) {
      return BuiltinTypes.SET;
    }
    if (predicate.test(BuiltinTypes.DICT)) {
      return BuiltinTypes.DICT;
    }
    if (predicate.test(BuiltinTypes.TUPLE)) {
      return BuiltinTypes.TUPLE;
    }
    return null;
  }

  private static boolean isException(InferredType inferredType) {
    return inferredType.canBeOrExtend("unittest.mock.Mock");
  }
}
