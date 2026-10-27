/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.matchers.InternalTypeMatchers;

import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_GROUP_CLAUSE;

@Rule(key = "S5708")
public class CaughtExceptionsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Change this expression to be a class deriving from BaseException or a tuple of such classes.";
  public static final String QUICK_FIX_MESSAGE_FORMAT = "Make \"%s\" deriving from \"Exception\"";
  private static final TypeMatcher BASE_EXCEPTION_MATCHER = TypeMatchers.any(
    TypeMatchers.isOrExtendsType("builtins.BaseException"),
    TypeMatchers.isObjectInstanceOf("builtins.BaseException"));
  private static final TypeMatcher TUPLE_MATCHER = TypeMatchers.isObjectOfType("tuple");
  private static final TypeMatcher RUNTIME_CLASS_OBJECT_MATCHER = TypeMatchers.isObjectOfType("type");
  private static final TypeMatcher VALID_HANDLER_MATCHER = TypeMatchers.any(
    BASE_EXCEPTION_MATCHER,
    TUPLE_MATCHER);
  private static final TypeMatcher POSSIBLY_VALID_UNION_HANDLER_MATCHER = InternalTypeMatchers.isAnyTypeInUnionSatisfying(VALID_HANDLER_MATCHER);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(EXCEPT_CLAUSE, CaughtExceptionsCheck::checkExceptClause);
    context.registerSyntaxNodeConsumer(EXCEPT_GROUP_CLAUSE, CaughtExceptionsCheck::checkExceptClause);
  }

  private static void checkExceptClause(SubscriptionContext ctx) {
    Expression exception = ((ExceptClause) ctx.syntaxNode()).exception();
    if (exception == null) {
      return;
    }

    TreeUtils.flattenTuples(exception).forEach(expression -> {
      if (!canBeOrExtendBaseException(expression, ctx) && !isAliasOfClass(expression, ctx)) {
        var issue = ctx.addIssue(expression, MESSAGE);
        addQuickFix(issue, expression);
      }
    });
  }

  private static void addQuickFix(PreciseIssue issue, SymbolV2 symbol) {
    symbol.usages()
      .stream()
      .filter(UsageV2::isBindingUsage)
      .findFirst()
      .map(UsageV2::tree)
      .map(Tree::parent)
      .map(TreeUtils.toInstanceOfMapper(ClassDef.class))
      .ifPresent(classDef -> addExceptionQuickFix(issue, classDef));
  }

  private static void addQuickFix(PreciseIssue issue, Expression expression) {
    SymbolV2 symbolV2 = symbolV2(expression);
    if (symbolV2 != null) {
      addQuickFix(issue, symbolV2);
    } else if (expression instanceof QualifiedExpression qualifiedExpression) {
      Symbol symbol = qualifiedExpression.symbol();
      if (symbol != null) {
        addQuickFix(issue, symbol);
      }
    }
  }

  private static void addQuickFix(PreciseIssue issue, Symbol symbol) {
    symbol.usages()
      .stream()
      .filter(Usage::isBindingUsage)
      .findFirst()
      .map(Usage::tree)
      .map(Tree::parent)
      .map(TreeUtils.toInstanceOfMapper(ClassDef.class))
      .ifPresent(classDef -> addExceptionQuickFix(issue, classDef));
  }

  private static void addExceptionQuickFix(PreciseIssue issue, ClassDef classDef) {
    Tree insertAfter = classDef.name();
    String insertingText = "(Exception)";

    Token leftPar = classDef.leftPar();
    if (leftPar != null) {
      ArgList args = classDef.args();
      if (args == null) {
        insertAfter = leftPar;
        insertingText = "Exception";
      } else {
        insertAfter = args;
        insertingText = ", Exception";
      }
    }

    issue.addQuickFix(PythonQuickFix.newQuickFix(String.format(QUICK_FIX_MESSAGE_FORMAT, classDef.name().name()))
      .addTextEdit(TextEditUtils.insertAfter(insertAfter, insertingText))
      .build());
  }

  private static boolean canBeOrExtendBaseException(Expression expression, SubscriptionContext ctx) {
    if (expression.typeV2() instanceof FunctionType || expression.typeV2() instanceof ModuleType) {
      return true;
    }
    return VALID_HANDLER_MATCHER.evaluateFor(expression, ctx) != TriBool.FALSE
      || RUNTIME_CLASS_OBJECT_MATCHER.evaluateFor(expression, ctx) == TriBool.TRUE;
  }

  private static SymbolV2 symbolV2(Expression expression) {
    if (expression instanceof Name name) {
      return name.symbolV2();
    }
    if (expression instanceof QualifiedExpression qualifiedExpression) {
      return qualifiedExpression.name().symbolV2();
    }
    return null;
  }

  private static boolean isAliasOfClass(Expression expression, SubscriptionContext ctx) {
    if (!(expression instanceof Name name)) {
      return false;
    }
    SymbolV2 symbol = name.symbolV2();
    if (symbol == null) {
      return false;
    }
    var bindingUsages = symbol.usages().stream().filter(UsageV2::isBindingUsage).toList();
    boolean isPotentiallyValidUnionHandler = !(name.typeV2() instanceof UnionType)
      || POSSIBLY_VALID_UNION_HANDLER_MATCHER.evaluateFor(name, ctx) != TriBool.FALSE;
    return (bindingUsages.stream().anyMatch(usage -> usage.kind() == UsageV2.Kind.PARAMETER)
      && !(name.typeV2() instanceof ObjectType objectType && objectType.type() instanceof ClassType)
      && isPotentiallyValidUnionHandler)
      || (name.typeV2() instanceof ClassType && bindingUsages.stream().anyMatch(usage -> usage.kind() == UsageV2.Kind.ASSIGNMENT_LHS))
      || (name.typeV2() instanceof UnionType && bindingUsages.size() > 1 && isPotentiallyValidUnionHandler);
  }
}
