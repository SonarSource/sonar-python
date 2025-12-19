/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7504")
public class UnnecessaryListCastCheck extends PythonSubscriptionCheck {
  private static final TypeMatcher isListCallMatcher = TypeMatchers.isType("list");
  private static final TypeMatcher modifyingListMethodMatcher = TypeMatchers.any(
      TypeMatchers.withFQN("list.append"),
      TypeMatchers.withFQN("list.extend"),
      TypeMatchers.withFQN("list.insert"),
      TypeMatchers.withFQN("list.remove"),
      TypeMatchers.withFQN("list.pop"),
      TypeMatchers.withFQN("typing.MutableSequence.clear"),
      TypeMatchers.withFQN("list.sort"),
      TypeMatchers.withFQN("typing.MutableSequence.reverse")
      );
  private static final TypeMatcher modifyingDictMethodMatcher = TypeMatchers.any(
    TypeMatchers.withFQN("dict.pop"),
    TypeMatchers.withFQN("typing.MutableMapping.popitem"),
    TypeMatchers.withFQN("typing.MutableMapping.clear"),
    TypeMatchers.withFQN("typing.MutableMapping.update"),
    TypeMatchers.withFQN("typing.MutableMapping.setdefault"));
  private static TypeMatcher dictViewMethodMatcher = TypeMatchers.any(
    TypeMatchers.withFQN("dict.keys"),
    TypeMatchers.withFQN("dict.items"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, UnnecessaryListCastCheck::checkForStatements);
    context.registerSyntaxNodeConsumer(Tree.Kind.COMP_FOR, UnnecessaryListCastCheck::checkComprehensions);
  }

  private static void checkForStatements(SubscriptionContext ctx) {
    ForStatement stmt = ((ForStatement) ctx.syntaxNode());
    hasListCallOnIterable(ctx, stmt.testExpressions())
      .filter(listCall -> !isListModifiedInLoop(ctx, listCall, stmt))
      .ifPresent(listCall -> raiseIssue(ctx, listCall));
  }

  private static void checkComprehensions(SubscriptionContext ctx) {
    ComprehensionFor comprehensionFor = ((ComprehensionFor) ctx.syntaxNode());
    hasListCallOnIterable(ctx, List.of(comprehensionFor.iterable()))
      .ifPresent(listCall -> raiseIssue(ctx, listCall));
  }

  private static Optional<CallExpression> hasListCallOnIterable(SubscriptionContext ctx, List<Expression> testExpressions) {
    if (testExpressions.size() == 1
      && testExpressions.get(0) instanceof CallExpression callExpression
      && isListCall(ctx, callExpression)
      && getFirstAndOnlyRegularArgument(callExpression).isPresent()) {
      return Optional.of(callExpression);
    }
    return Optional.empty();
  }

  private static boolean isListCall(SubscriptionContext ctx, CallExpression callExpression) {
    return isListCallMatcher.isTrueFor(callExpression.callee(), ctx);
  }

  private static boolean isListModifiedInLoop(SubscriptionContext ctx, CallExpression callExpression, ForStatement forStatement) {
    var listName = getFirstNameArgument(callExpression);
    if (listName.isPresent()) {
      ModifyingCollectionTreeVisitor visitor = new ModifyingCollectionTreeVisitor(ctx, listName.get(),
        modifyingListMethodMatcher, false);
      forStatement.accept(visitor);
      return visitor.isModifyingCollection();
    }

    var dictNameAndMethod = getDictNameFromViewCall(ctx, callExpression);
    if (dictNameAndMethod.isPresent()) {
      ModifyingCollectionTreeVisitor visitor = new ModifyingCollectionTreeVisitor(ctx, dictNameAndMethod.get(),
        modifyingDictMethodMatcher, true);
      forStatement.accept(visitor);
      return visitor.isModifyingCollection();
    }

    return false;
  }

  public static Optional<Name> getFirstNameArgument(CallExpression callExpression) {
    return getFirstAndOnlyRegularArgument(callExpression)
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class));
  }

  /**
   * Extracts the dict name from list(dict.keys()) or list(dict.items()) calls
   */
  private static Optional<Name> getDictNameFromViewCall(SubscriptionContext ctx, CallExpression callExpression) {
    return getFirstAndOnlyRegularArgument(callExpression)
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .filter(innerCall -> isDictViewCall(ctx, innerCall))
      .map(CallExpression::callee)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::qualifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class));
  }

  private static boolean isDictViewCall(SubscriptionContext ctx, CallExpression callExpr) {
    return dictViewMethodMatcher.isTrueFor(callExpr.callee(), ctx);
  }

  private static Optional<RegularArgument> getFirstAndOnlyRegularArgument(CallExpression callExpression) {
    if (callExpression.arguments().size() == 1
      && callExpression.arguments().get(0) instanceof RegularArgument regularArgument) {
      return Optional.of(regularArgument);
    }
    return Optional.empty();
  }

  private static void raiseIssue(SubscriptionContext ctx, CallExpression listCall) {
    PreciseIssue issue = ctx.addIssue(listCall.callee(),
      "Remove this unnecessary `list()` call on an already iterable object.");
    Optional.ofNullable(listCall.argumentList())
      .map(argList -> TreeUtils.treeToString(argList, false))
      .map(replacementText -> TextEditUtils.replace(listCall, replacementText))
      .map(textEdit -> PythonQuickFix.newQuickFix("Remove the \"list\" call", textEdit))
      .ifPresent(issue::addQuickFix);
  }

  private static class ModifyingCollectionTreeVisitor extends BaseTreeVisitor {
    private final SubscriptionContext ctx;
    private final Name collectionName;
    private final TypeMatcher methodsMatcher;
    private final boolean isDict;

    private boolean isModifyingCollection = false;

    ModifyingCollectionTreeVisitor(SubscriptionContext ctx, Name collectionName, TypeMatcher methodsMatcher,
      boolean isDict) {
      this.ctx = ctx;
      this.collectionName = collectionName;
      this.methodsMatcher = methodsMatcher;
      this.isDict = isDict;
    }

    public boolean isModifyingCollection() {
      return isModifyingCollection;
    }

    @Override
    public void visitCallExpression(CallExpression callExpr) {
      super.visitCallExpression(callExpr);
      var symbol = collectionName.symbolV2();
      if (symbol != null) {
        isModifyingCollection |= 
          isMethodReceiverInstanceOf(callExpr, symbol) && isModifyingMethod(callExpr);
      }
    }

    @Override
    public void visitDelStatement(DelStatement delStatement) {
      super.visitDelStatement(delStatement);
      if (isDict) {
        var symbol = collectionName.symbolV2();
        if (symbol != null) {
          // Check if any of the deleted expressions is a subscription of our dict
          isModifyingCollection |= delStatement.expressions().stream()
            .anyMatch(expr -> isSubscriptionOf(expr, symbol));
        }
      }
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement assignmentStatement) {
      super.visitAssignmentStatement(assignmentStatement);
      if (isDict) {
        var symbol = collectionName.symbolV2();
        if (symbol != null) {
          isModifyingCollection |= assignmentStatement.lhsExpressions().stream()
            .flatMap(exprList -> exprList.expressions().stream())
            .anyMatch(expr -> isSubscriptionOf(expr, symbol));
        }
      }
    }

    private static boolean isMethodReceiverInstanceOf(CallExpression callExpr, SymbolV2 collectionSymbol) {
      if (callExpr.callee() instanceof QualifiedExpression qualifiedExpression &&
        qualifiedExpression.qualifier() instanceof Name name) {
        var symbolV2 = name.symbolV2();
        return symbolV2 != null && symbolV2.equals(collectionSymbol);
      }
      return false;
    }

    private static boolean isSubscriptionOf(Expression expr, SymbolV2 collectionSymbol) {
      if (expr instanceof SubscriptionExpression subscriptionExpr &&
        subscriptionExpr.object() instanceof Name name) {
        var symbolV2 = name.symbolV2();
        return symbolV2 != null && symbolV2.equals(collectionSymbol);
      }
      return false;
    }

    private boolean isModifyingMethod(CallExpression callExpression) {
      return methodsMatcher.isTrueFor(callExpression.callee(), ctx);
    }
  }

}
