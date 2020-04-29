/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.SymbolUtils;

// https://jira.sonarsource.com/browse/SONARPY-668
// https://jira.sonarsource.com/browse/RSPEC-5792
@Rule(key = "S5792")
public class CsrfDisabledCheck extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, CsrfDisabledCheck::djangoMiddlewareArrayCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, CsrfDisabledCheck::djangoCsrfExemptCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, CsrfDisabledCheck::flaskWtfCsrfEnabledFalseCheck);
  }

  private static final String CSRF_VIEW_MIDDLEWARE = "django.middleware.csrf.CsrfViewMiddleware";

  /** Checks that <code>django.middleware.csrf.CsrfViewMiddleware</code> is in <code>MIDDLEWARE</code> array. */
  private static void djangoMiddlewareArrayCheck(SubscriptionContext subscriptionContext) {
    if (!"settings.py".equals(subscriptionContext.pythonFile().fileName())) {
      return;
    }

    AssignmentStatement asgn = (AssignmentStatement) subscriptionContext.syntaxNode();
    // Check that the left hand side is called `MIDDLEWARE` and that there is at least one string entry starting with
    // "django" in that array.
    boolean isLhsCalledMiddleware = isLhsCalled("MIDDLEWARE").test(asgn);
    boolean containsDjangoMiddleware = isListAnyMatch(isString(s -> s.startsWith("django"))).test(asgn.assignedValue());
    boolean isMiddlewareAssignment = isLhsCalledMiddleware && containsDjangoMiddleware;
    if (isMiddlewareAssignment) {
      boolean containsCsrfViewMiddleware = isListAnyMatch(isString(CSRF_VIEW_MIDDLEWARE)).test(asgn.assignedValue());
      if (!containsCsrfViewMiddleware) {
        subscriptionContext.addIssue(
          asgn.lastToken(),
          "CSRF protection (django.middleware.csrf.CsrfViewMiddleware) is missing.");
      }
    }
  }

  /** Checks that the left hand side of the assignment is a variable with name <code>lhsName</code>. */
  private static Predicate<AssignmentStatement> isLhsCalled(String lhsName) {
    return asgn -> asgn.lhsExpressions().stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .filter(expr -> expr.is(Tree.Kind.NAME) && lhsName.equals(((Name) expr).name()))
      .findFirst()
      .isPresent();
  }

  /** Checks whether an expression is a string literal satisfying a predicate. */
  private static Predicate<Expression> isString(Predicate<String> pred) {
    return expr -> {
      if (expr.is(Tree.Kind.STRING_LITERAL)) {
        List<StringElement> elems = ((StringLiteral) expr).stringElements();
        return elems.size() == 1 && pred.test(trimQuotes(elems.get(0).value()));
      } else {
        return false;
      }
    };
  }

  private static String trimQuotes(String s) {
    if ((s.startsWith("'") && s.endsWith("'")) || (s.startsWith("\"") && s.endsWith("\""))) {
      return s.substring(1, s.length() - 1);
    } else {
      return s;
    }
  }

  /** Checks whether the expression is a string literal with value exactly equal to <code>s</code>. */
  private static Predicate<Expression> isString(String s) {
    return isString(s::equals);
  }

  /** Checks that an expression is a list literal with at least one entry satisfying the predicate. */
  private static Predicate<Expression> isListAnyMatch(Predicate<Expression> pred) {
    return expr -> Optional.ofNullable(expr)
      .filter(ListLiteral.class::isInstance)
      .flatMap(lst-> ((ListLiteral) lst).elements().expressions().stream().filter(pred).findFirst())
      .isPresent();
  }

  /** Raises issue whenever <code>@csrf_exempt</code> decorator is found. */
  private static void djangoCsrfExemptCheck(SubscriptionContext subscriptionContext) {
    Decorator decorator = (Decorator) subscriptionContext.syntaxNode();
    if (decorator.name().names().size() == 1 && "csrf_exempt".equals(decorator.name().names().get(0).name())) {
      subscriptionContext.addIssue(decorator.lastToken(), "Disabling CSRF protection is dangerous.");
    }
  }

  private static void flaskWtfCsrfEnabledFalseCheck(SubscriptionContext subscriptionContext) {
    AssignmentStatement asgn = (AssignmentStatement) subscriptionContext.syntaxNode();
    // Checks that the left hand side is some kind of subscription of `something['WTF_CSRF_ENABLED']`;
    // Does not check what `something` is - overtainting seems extremely unlikely in this case.
    boolean isWtfCsrfEnabledSubscription = asgn
      .lhsExpressions()
      .stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .filter(expr -> expr.is(Tree.Kind.SUBSCRIPTION))
      .flatMap(s -> ((SubscriptionExpression) s).subscripts().expressions().stream())
      .filter(isString("WTF_CSRF_ENABLED"))
      .findFirst()
      .isPresent();
    if (isWtfCsrfEnabledSubscription && Expressions.isFalsy(asgn.assignedValue())) {
      subscriptionContext.addIssue(asgn.assignedValue(), "Disabling CSRF protection is dangerous.");
    }
  }

}
