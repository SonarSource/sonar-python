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
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

@Rule(key = "S7504")
public class UnnecessaryListCastCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder isListCallCheck;

  private static final Set<String> MODIFYING_LIST_METHODS = Set.of("append", "extend", "insert", "remove", "pop",
      "clear", "sort", "reverse");
  private TypeCheckMap<Object> typeCheckMap;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, this::checkForStatements);
    context.registerSyntaxNodeConsumer(Tree.Kind.COMP_FOR, this::checkComprehensions);
  }

  private void initChecks(SubscriptionContext ctx) {
    isListCallCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("list");

    var marker = new Object();
    typeCheckMap = new TypeCheckMap<>();
    MODIFYING_LIST_METHODS.forEach(method -> {
      var checker = ctx.typeChecker().typeCheckBuilder().isTypeWithName("list." + method);
      typeCheckMap.put(checker, marker);
    });
  }

  private void checkForStatements(SubscriptionContext ctx) {
    ForStatement stmt = ((ForStatement) ctx.syntaxNode());
    hasListCallOnIterable(stmt.testExpressions())
        .filter(listCall -> !isListModifiedInLoop(listCall, stmt))
        .ifPresent(listCall -> raiseIssue(ctx, listCall));
  }

  private void checkComprehensions(SubscriptionContext ctx) {
    ComprehensionFor comprehensionFor = ((ComprehensionFor) ctx.syntaxNode());
    hasListCallOnIterable(List.of(comprehensionFor.iterable()))
        .ifPresent(listCall -> raiseIssue(ctx, listCall));
  }

  private Optional<CallExpression> hasListCallOnIterable(List<Expression> testExpressions) {
    if (testExpressions.size() == 1
        && testExpressions.get(0) instanceof CallExpression callExpression
        && isListCall(callExpression)
        && getFirstRegularArgument(callExpression).isPresent()) {
      return Optional.of(callExpression);
    }
    return Optional.empty();
  }

  private boolean isListCall(CallExpression callExpression) {
    return isListCallCheck.check(callExpression.callee().typeV2()) == TriBool.TRUE;
  }

  private boolean isListModifiedInLoop(CallExpression callExpression, ForStatement forStatement) {
    var listName = getFirstNameArgument(callExpression);
    return listName.map(name -> {
      ModifyingListMethodTreeVisitor visitor = new ModifyingListMethodTreeVisitor(name, typeCheckMap);
      forStatement.accept(visitor);
      return visitor.isModifyingListMethod();
    }).orElse(false);
  }

  public static Optional<Name> getFirstNameArgument(CallExpression callExpression) {
    return getFirstRegularArgument(callExpression)
        .map(RegularArgument::expression)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class));
  }

  private static Optional<RegularArgument> getFirstRegularArgument(CallExpression callExpression) {
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

  private static class ModifyingListMethodTreeVisitor extends BaseTreeVisitor {
    private final Name listName;
    private final TypeCheckMap<Object> modifyingListTypeCheckMap;

    private boolean isModifyingListMethod = false;

    ModifyingListMethodTreeVisitor(Name listName, TypeCheckMap<Object> modifyingListTypeCheckMap) {
      this.listName = listName;
      this.modifyingListTypeCheckMap = modifyingListTypeCheckMap;
    }

    public boolean isModifyingListMethod() {
      return isModifyingListMethod;
    }

    @Override
    public void visitCallExpression(CallExpression callExpr) {
      super.visitCallExpression(callExpr);
      var symbol = listName.symbolV2();
      if (symbol != null) {
        isModifyingListMethod |= isMethodReceiverInstanceOf(callExpr, symbol)
            && isModifyingListMethod(callExpr);
      }
    }

    private static boolean isMethodReceiverInstanceOf(CallExpression callExpr, SymbolV2 listSymbol) {
      if (callExpr.callee() instanceof QualifiedExpression qualifiedExpression &&
          qualifiedExpression.qualifier() instanceof Name name) {
        var symbolV2 = name.symbolV2();
        return symbolV2 != null && symbolV2.equals(listSymbol);
      }
      return false;
    }

    private boolean isModifyingListMethod(CallExpression callExpression) {
      return modifyingListTypeCheckMap.containsForType(callExpression.callee().typeV2());
    }
  }

}
