/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.checks;

import java.util.Collection;
import java.util.Map;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.InExpression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

import static org.sonar.python.tree.TreeUtils.nameFromExpression;
import static org.sonar.python.types.InferredTypes.containsDeclaredType;
import static org.sonar.python.types.InferredTypes.typeClassLocation;
import static org.sonar.python.types.InferredTypes.typeName;
import static org.sonar.python.types.InferredTypes.typeSymbols;

@Rule(key = "S5864")
public class ConfusingTypeCheckingCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    new NonCallableCalledCheck().initialize(context);
    new IncompatibleOperandsCheck().initialize(context);
    new ItemOperationsTypeCheck().initialize(context);
    new IterationOnNonIterableCheck().initialize(context);
    new SillyEqualityCheck().initialize(context);
    context.registerSyntaxNodeConsumer(Tree.Kind.RAISE_STMT, ConfusingTypeCheckingCheck::checkIncorrectExceptionType);
    context.registerSyntaxNodeConsumer(Tree.Kind.IS, ConfusingTypeCheckingCheck::checkSillyIdentity);
  }

  private static class IncompatibleOperandsCheck extends IncompatibleOperands {
    @Override
    public SpecialMethod resolveMethod(InferredType type, String method) {
      Symbol symbol = type.resolveDeclaredMember(method).orElse(null);
      boolean isUnresolved = !containsDeclaredType(type) || (symbol == null && type.declaresMember(method));
      return new SpecialMethod(symbol, isUnresolved);
    }

    @Override
    public String message(Token operator) {
      return "Fix this \"" + operator.value() + "\" operation; Previous type checks suggest that operand has incompatible type.";
    }

    @Override
    public String message(Token operator, InferredType left, InferredType right) {
      String leftTypeName = InferredTypes.typeName(left);
      String rightTypeName = InferredTypes.typeName(right);
      String message = "Fix this \"" + operator.value() + "\" operation; Previous type checks suggest that operands have incompatible types";
      if (leftTypeName != null && rightTypeName != null) {
        message += " (" + leftTypeName + " and " + rightTypeName + ")";
      }
      return message + ".";
    }
  }

  private static class ItemOperationsTypeCheck extends ItemOperationsType {

    @Override
    public boolean isValidSubscription(Expression subscriptionObject, String requiredMethod, @Nullable String classRequiredMethod,
                                       Map<LocationInFile, String> secondaries) {

      InferredType type = subscriptionObject.type();
      String typeName = InferredTypes.typeName(type);
      String secondaryMessage = typeName != null ? String.format(SECONDARY_MESSAGE, typeName) : DEFAULT_SECONDARY_MESSAGE;
      secondaries.put(typeClassLocation(type), secondaryMessage);
      if (!containsDeclaredType(type)) {
        // handled by S5644
        return true;
      }
      return isUsedInsideInExpression(subscriptionObject) || type.declaresMember(requiredMethod);
    }

    private static boolean isUsedInsideInExpression(Expression subscriptionObject) {
      return TreeUtils.getSymbolFromTree(subscriptionObject)
        .filter(symbol -> symbol.usages().stream().anyMatch(usage -> isRightOperandInExpression(usage.tree())))
        .isPresent();
    }

    private static boolean isRightOperandInExpression(Tree tree) {
      Tree parent = tree.parent();
      if (parent instanceof InExpression inExpression) {
        return inExpression.rightOperand() == tree;
      }
      return false;
    }

    @Override
    public String message(@Nullable String name, String missingMethod) {
      if (name != null) {
        return String.format("Fix this \"%s\" operation; Previous type checks suggest that \"%s\" does not have this method.", missingMethod, name);
      }
      return String.format("Fix this \"%s\" operation; Previous type checks suggest that this expression does not have this method.", missingMethod);
    }
  }

  private static class IterationOnNonIterableCheck extends IterationOnNonIterable {

    @Override
    boolean isValidIterable(Expression expression, Map<LocationInFile, String> secondaries) {
      InferredType type = expression.type();
      String typeName = InferredTypes.typeName(type);
      String secondaryMessage = typeName != null ? String.format(SECONDARY_MESSAGE, typeName) : DEFAULT_SECONDARY_MESSAGE;
      secondaries.put(InferredTypes.typeClassLocation(type), secondaryMessage);
      return !containsDeclaredType(type) || type.declaresMember("__iter__") || type.declaresMember("__getitem__");
    }

    @Override
    String message(Expression expression, boolean isForLoop) {
      String typeName = InferredTypes.typeName(expression.type());
      String expressionName = nameFromExpression(expression);
      String expressionNameString = expressionName != null ? String.format("\"%s\"", expressionName) : "it";
      String typeNameString = typeName != null ? String.format("has type \"%s\" and", typeName) : "";
      return isForLoop && isAsyncIterable(expression) ?
        String.format("Add \"async\" before \"for\"; Previous type checks suggest that %s %s is an async generator.", expressionNameString, typeNameString) :
        String.format("Replace this expression; Previous type checks suggest that %s %s isn't iterable.", expressionNameString, typeNameString);
    }

    @Override
    boolean isAsyncIterable(Expression expression) {
      InferredType type = expression.type();
      // No need to check again if the type contains a declared type
      return type.declaresMember("__aiter__");
    }
  }

  private static void checkIncorrectExceptionType(SubscriptionContext ctx) {
    RaiseStatement raiseStatement = (RaiseStatement) ctx.syntaxNode();
    if (raiseStatement.expressions().isEmpty()) {
      return;
    }
    Expression raisedExpression = raiseStatement.expressions().get(0);
    InferredType type = raisedExpression.type();
    if (!containsDeclaredType(type)) {
      return;
    }
    if (!type.isCompatibleWith(InferredTypes.runtimeType(TypeShed.typeShedClass("BaseException")))) {
      String expressionName = nameFromExpression(raisedExpression) != null ? String.format("\"%s\"", nameFromExpression(raisedExpression)) : "this expression";
      String typeName = typeName(type);
      ctx.addIssue(raiseStatement, String.format("Fix this \"raise\" statement; Previous type checks suggest that %s has type \"%s\" and is not an exception.",
        expressionName, typeName));
    }
  }

  private static class SillyEqualityCheck extends SillyEquality {

    @Override
    boolean areIdentityComparableOrNone(InferredType leftType, InferredType rightType) {
      return ConfusingTypeCheckingCheck.areIdentityComparableOrNone(leftType, rightType);
    }

    @Override
    public boolean canImplementEqOrNe(Expression expression) {
      InferredType type = expression.type();
      // inferredType will always contain a declared type because of the check done inside 'areIdentityComparableOrNone'
      return type.declaresMember("__eq__") || type.declaresMember("__ne__");
    }

    @CheckForNull
    @Override
    String builtinTypeCategory(InferredType inferredType) {
      // inferredType will always contain a declared type because of the check done inside 'areIdentityComparableOrNone'
      Map<String, String> builtinsTypeCategory = InferredTypes.getBuiltinsTypeCategory();
      return builtinsTypeCategory.keySet().stream()
        .filter(typeName -> typeSymbols(inferredType).stream().map(Symbol::fullyQualifiedName).allMatch(typeName::equals))
        .map(builtinsTypeCategory::get).findFirst().orElse(null);
    }

    @Override
    String message(String result) {
      return "Fix this equality check; Previous type checks suggest that operands have incompatible types.";
    }
  }

  private static class NonCallableCalledCheck extends NonCallableCalled {

    @Override
    protected boolean isExpectedTypeSource(SubscriptionContext ctx, PythonType calleeType) {
      return ctx.typeChecker().typeCheckBuilder().isTypeHintTypeSource().check(calleeType) == TriBool.TRUE;
    }

    @Override
    protected boolean isException(SubscriptionContext ctx, PythonType calleeType) {
      var isCoroutine = ctx.typeChecker().typeCheckBuilder().isInstanceOf("typing.Coroutine").check(calleeType) == TriBool.TRUE;
      return super.isException(ctx, calleeType) || isCoroutine;
    }
  }

  private static boolean areIdentityComparableOrNone(InferredType leftType, InferredType rightType) {
    if (!containsDeclaredType(leftType) && !containsDeclaredType(rightType)) {
      return true;
    }
    Collection<ClassSymbol> leftSymbols = typeSymbols(leftType);
    Collection<ClassSymbol> rightSymbols = typeSymbols(rightType);
    return leftSymbols.stream().map(Symbol::fullyQualifiedName).anyMatch(fqn -> fqn == null || rightSymbols.stream().anyMatch(rs -> rs.canBeOrExtend(fqn)))
      || rightSymbols.stream().map(Symbol::fullyQualifiedName).anyMatch(fqn -> fqn == null || leftSymbols.stream().anyMatch(ls -> ls.canBeOrExtend(fqn)))
      || (typeSymbols(leftType).stream().map(Symbol::fullyQualifiedName).allMatch("NoneType"::equals))
      || (typeSymbols(rightType).stream().map(Symbol::fullyQualifiedName).allMatch("NoneType"::equals));
  }

  private static void checkSillyIdentity(SubscriptionContext ctx) {
    IsExpression isExpression = (IsExpression) ctx.syntaxNode();
    InferredType left = isExpression.leftOperand().type();
    InferredType right = isExpression.rightOperand().type();
    if (!areIdentityComparableOrNone(left, right)) {
      Token notToken = isExpression.notToken();
      Token lastToken = notToken == null ? isExpression.operator() : notToken;
      ctx.addIssue(isExpression.operator(), lastToken, "Fix this identity check; Previous type checks suggest that operands have incompatible types.");
    }
  }
}
