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
package org.sonar.python.checks.hotspots;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.SymbolUtils;

import static org.sonar.python.checks.Expressions.isFalsy;

@Rule(key = "S2092")
public class SecureCookieCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Make sure creating this cookie without the \"secure\" flag is safe.";
  private static final Set<String> SET_COOKIE_METHODS = new HashSet<>(Arrays.asList(
    "django.http.HttpResponse.set_cookie",
    "django.http.HttpResponse.set_signed_cookie",
    "django.http.HttpResponseRedirect.set_cookie",
    "django.http.HttpResponseRedirect.set_signed_cookie",
    "django.http.HttpResponsePermanentRedirect.set_cookie",
    "django.http.HttpResponsePermanentRedirect.set_signed_cookie",
    "django.http.HttpResponseNotModified.set_cookie",
    "django.http.HttpResponseNotModified.set_signed_cookie",
    "django.http.HttpResponseBadRequest.set_cookie",
    "django.http.HttpResponseBadRequest.set_signed_cookie",
    "django.http.HttpResponseNotFound.set_cookie",
    "django.http.HttpResponseNotFound.set_signed_cookie",
    "django.http.HttpResponseForbidden.set_cookie",
    "django.http.HttpResponseForbidden.set_signed_cookie",
    "django.http.HttpResponseNotAllowed.set_cookie",
    "django.http.HttpResponseNotAllowed.set_signed_cookie",
    "django.http.HttpResponseGone.set_cookie",
    "django.http.HttpResponseGone.set_signed_cookie",
    "django.http.HttpResponseServerError.set_cookie",
    "django.http.HttpResponseServerError.set_signed_cookie"
  ));

  private static final String SECURE = "secure";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> {
      AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
      getSubscriptionToCookies(assignment.lhsExpressions())
        .forEach(sub -> {
          if (isSettingSecureFlag(sub) && isFalsy(assignment.assignedValue())) {
            ctx.addIssue(assignment, MESSAGE);
          }
        });
    });

    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && SET_COOKIE_METHODS.contains(calleeSymbol.fullyQualifiedName())) {
        if (callExpression.arguments().stream().anyMatch(argument -> argument.is(Kind.UNPACKING_EXPR))) {
          return;
        }
        RegularArgument secureArgument = getSecureArgument(callExpression.arguments());
        if (secureArgument == null || isFalsy(secureArgument.expression())) {
          ctx.addIssue(callExpression.callee(), MESSAGE);
        }
      }
    });
  }

  @CheckForNull
  private static RegularArgument getSecureArgument(List<Argument> arguments) {
    return arguments.stream()
      .map(RegularArgument.class::cast)
      .filter(argument -> {
        Name keywordArgument = argument.keywordArgument();
        return keywordArgument != null && keywordArgument.name().equals(SECURE);
      })
      .findFirst().orElse(null);
  }

  private static Stream<SubscriptionExpression> getSubscriptionToCookies(List<ExpressionList> lhsExpressions) {
    return lhsExpressions.stream()
      .flatMap(expressionList -> expressionList.expressions().stream())
      .filter(lhs -> {
        if (lhs.is(Kind.SUBSCRIPTION)) {
          SubscriptionExpression sub = (SubscriptionExpression) lhs;
          Symbol objectSymbol = getObjectSymbol(sub.object());
          return "http.cookies.SimpleCookie".equals(SymbolUtils.getTypeName(objectSymbol));
        }
        return false;
      })
      .map(SubscriptionExpression.class::cast);
  }

  private static boolean isSettingSecureFlag(SubscriptionExpression sub) {
    List<ExpressionList> subscripts = getSubscripts(sub);
    if (subscripts.size() == 1) {
      return false;
    }
    return subscripts.stream()
      .skip(1)
      .anyMatch(s -> s.expressions().size() == 1 && isSecureStringLiteral(s.expressions().get(0)));
  }

  private static List<ExpressionList> getSubscripts(SubscriptionExpression sub) {
    Deque<ExpressionList> subscripts = new ArrayDeque<>();
    subscripts.addFirst(sub.subscripts());
    Expression object = sub.object();
    while (object.is(Kind.SUBSCRIPTION)) {
      subscripts.addFirst(((SubscriptionExpression) object).subscripts());
      object = ((SubscriptionExpression) object).object();
    }
    return new ArrayList<>(subscripts);
  }

  private static boolean isSecureStringLiteral(Expression expression) {
    return expression.is(Kind.STRING_LITERAL) && ((StringLiteral) expression).trimmedQuotesValue().equalsIgnoreCase(SECURE);
  }

  @CheckForNull
  private static Symbol getObjectSymbol(Expression object) {
    if (object.is(Kind.SUBSCRIPTION)) {
      return getObjectSymbol(((SubscriptionExpression) object).object());
    }
    if (object instanceof HasSymbol) {
      return ((HasSymbol) object).symbol();
    }
    return null;
  }

}
