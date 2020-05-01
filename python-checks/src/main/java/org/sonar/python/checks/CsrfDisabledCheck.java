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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;

// https://jira.sonarsource.com/browse/SONARPY-668
// https://jira.sonarsource.com/browse/RSPEC-5792
@Rule(key = "S5792")
public class CsrfDisabledCheck extends PythonSubscriptionCheck {

  private static final String DISABLING_CSRF_MESSAGE = "Disabling CSRF protection is dangerous.";
  private static final String CSRFPROTECT_MISSING_MESSAGE = "CSRFProtect is missing.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, CsrfDisabledCheck::djangoMiddlewareArrayCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, CsrfDisabledCheck::decoratorCsrfExemptCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, CsrfDisabledCheck::functionCsrfExemptCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, CsrfDisabledCheck::flaskWtfCsrfEnabledFalseCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, CsrfDisabledCheck::metaCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, CsrfDisabledCheck::formInstantiationCheck);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, CsrfDisabledCheck::improperlyConfiguredFlaskApp);
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
      .anyMatch(expr -> expr.is(Tree.Kind.NAME) && lhsName.equals(((Name) expr).name()));
  }

  /** Checks whether an expression is a string literal satisfying a predicate. */
  private static Predicate<Expression> isString(Predicate<String> pred) {
    return expr -> {
      if (expr.is(Tree.Kind.STRING_LITERAL)) {
        List<StringElement> elems = ((StringLiteral) expr).stringElements();
        return elems.size() == 1 && pred.test(elems.get(0).trimmedQuotesValue());
      } else {
        return false;
      }
    };
  }

  /** Checks whether the expression is a string literal with value exactly equal to <code>s</code>. */
  private static Predicate<Expression> isString(String s) {
    return isString(s::equals);
  }

  /** Checks that an expression is a list literal with at least one entry satisfying the predicate. */
  private static Predicate<Expression> isListAnyMatch(Predicate<Expression> pred) {
    return expr -> Optional.ofNullable(expr)
      .filter(ListLiteral.class::isInstance)
      .flatMap(lst -> ((ListLiteral) lst).elements().expressions().stream().filter(pred).findFirst())
      .isPresent();
  }

  private static final Set<String> DANGEROUS_DECORATORS = new HashSet<>(Arrays.asList(
    "django.views.decorators.csrf.csrf_exempt",
    "flask_wtf.csrf.CSRFProtect.exempt"));

  /** Raises issue whenever a decorator with something about "CSRF" and "exempt" in the combined name is found. */
  private static void decoratorCsrfExemptCheck(SubscriptionContext subscriptionContext) {
    Decorator decorator = (Decorator) subscriptionContext.syntaxNode();
    List<String> names = decorator.name().names().stream().map(Name::name).collect(Collectors.toList());
    // This is a temporary workaround until symbol resolution works for decorators.
    // Use the actual functions with FQNs from DANGEROUS_DECORATORS once that's fixed.
    boolean isDangerous = names.stream().anyMatch(s -> s.toLowerCase(Locale.US).contains("csrf")) &&
      names.stream().anyMatch(s -> s.toLowerCase(Locale.US).contains("exempt"));
    if (isDangerous) {
      subscriptionContext.addIssue(decorator.lastToken(), DISABLING_CSRF_MESSAGE);
    }
  }

  /** Raises an issue whenever one of the CSRF-exemption decorators is used as an ordinary function. */
  private static void functionCsrfExemptCheck(SubscriptionContext subscriptionContext) {
    CallExpression callExpr = (CallExpression) subscriptionContext.syntaxNode();
    Optional.ofNullable(callExpr.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(DANGEROUS_DECORATORS::contains)
      .ifPresent(fqn -> subscriptionContext.addIssue(callExpr.callee().lastToken(), DISABLING_CSRF_MESSAGE));
  }

  /** Checks that <code>'WTF_CSRF_ENABLED'</code> setting is not switched off. */
  private static void flaskWtfCsrfEnabledFalseCheck(SubscriptionContext subscriptionContext) {
    AssignmentStatement asgn = (AssignmentStatement) subscriptionContext.syntaxNode();
    // Checks that the left hand side is some kind of subscription of `something['WTF_CSRF_ENABLED']`
    // Does not check what `something` is - overtainting seems extremely unlikely in this case.
    boolean isWtfCsrfEnabledSubscription = asgn
      .lhsExpressions()
      .stream()
      .flatMap(exprList -> exprList.expressions().stream())
      .filter(expr -> expr.is(Tree.Kind.SUBSCRIPTION))
      .flatMap(s -> ((SubscriptionExpression) s).subscripts().expressions().stream())
      .anyMatch(isString(s -> "WTF_CSRF_ENABLED".equals(s) || "WTF_CSRF_CHECK_DEFAULT".equals(s)));
    if (isWtfCsrfEnabledSubscription && Expressions.isFalsy(asgn.assignedValue())) {
      subscriptionContext.addIssue(asgn.assignedValue(), DISABLING_CSRF_MESSAGE);
    }
  }

  /**
   * Detects <code>class Meta</code> withing <code>FlaskForm</code>-subclasses,
   * with <code>csrf</code> set to <code>False</code>.
   */
  private static void metaCheck(SubscriptionContext subscriptionContext) {
    ClassDef classDef = (ClassDef) subscriptionContext.syntaxNode();
    boolean isMetaClass = "Meta".equals(classDef.name().name());
    if (!isMetaClass) {
      return;
    }

    boolean isWithinFlaskForm = Optional.ofNullable(classDef.parent())
      .map(Tree::parent)
      .flatMap(tryCast(ClassDef.class))
      .map(parentClassDef -> parentClassDef.name().symbol())
      .flatMap(tryCast(ClassSymbol.class))
      .filter(parentClassSymbol -> parentClassSymbol.canBeOrExtend("flask_wtf.FlaskForm"))
      .isPresent();
    if (!isWithinFlaskForm) {
      return;
    }

    classDef.body().statements().forEach(stmt -> tryCast(AssignmentStatement.class, stmt)
      .filter(isLhsCalled("csrf"))
      .filter(asgn -> Expressions.isFalsy(asgn.assignedValue()))
      .ifPresent(asgn -> subscriptionContext.addIssue(asgn.assignedValue(), DISABLING_CSRF_MESSAGE)));
  }

  /** Attempts to cast a value of type <code>A</code> as a <code>B</code>. */
  private static <A, B> Optional<B> tryCast(Class<B> c, A a) {
    return c.isInstance(a) ? Optional.of(c.cast(a)) : Optional.empty();
  }

  /** Curried version of 2-ary <code>tryCast</code>, to be used inside <code>Optional.flatMap</code>. */
  private static <A, B> Function<A, Optional<B>> tryCast(Class<B> c) {
    return a -> tryCast(c, a);
  }

  /** Checks that subclasses of <code>FlaskForm</code> are instantiated without bad CSRF settings. */
  private static void formInstantiationCheck(SubscriptionContext subscriptionContext) {
    CallExpression callExpr = (CallExpression) subscriptionContext.syntaxNode();
    boolean isFlaskFormInstantiation = Optional.ofNullable(callExpr.calleeSymbol())
      .flatMap(tryCast(ClassSymbol.class))
      .filter(c -> c.canBeOrExtend("flask_wtf.FlaskForm"))
      .isPresent();
    if (!isFlaskFormInstantiation) {
      return;
    }

    callExpr.arguments().forEach(arg -> {
      if (arg instanceof RegularArgument) {
        RegularArgument regArg = (RegularArgument) arg;
        searchForProblemsInFormInitializationArguments(regArg)
          .ifPresent(badExpr -> subscriptionContext.addIssue(badExpr, DISABLING_CSRF_MESSAGE));
      }
    });
  }

  /**
   * Attempts to find dangerous settings in a regular argument used in Flask form initialization.
   */
  private static Optional<Expression> searchForProblemsInFormInitializationArguments(RegularArgument regArg) {
    String name = Optional.ofNullable(regArg.keywordArgument()).map(Name::name).orElse(null);
    if ("csrf_enabled".equals(name) && Expressions.isFalsy(regArg.expression())) {
      return Optional.of(regArg.expression());
    } else if ("meta".equals(name)) {
      return tryCast(DictionaryLiteral.class, regArg.expression())
        .flatMap(CsrfDisabledCheck::searchForBadCsrfSettingInDictionary);
    } else {
      return Optional.empty();
    }
  }

  /** Looks for <code>'csrf': False</code> and similar settings in a dictionary. */
  private static Optional<Expression> searchForBadCsrfSettingInDictionary(DictionaryLiteral dict) {
    return dict.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .filter(kvp -> tryCast(StringLiteral.class, kvp.key())
        .filter(strLit -> "csrf".equals(strLit.trimmedQuotesValue()))
        .isPresent())
      .findFirst()
      .filter(kvp -> Expressions.isFalsy(kvp.value()))
      .map(KeyValuePair::value);
  }

  private static void improperlyConfiguredFlaskApp(SubscriptionContext subscriptionContext) {
    AssignmentStatement asgn = (AssignmentStatement) subscriptionContext.syntaxNode();
    if (isFlaskAppInstantiation(asgn.assignedValue())) {
      boolean isCsrfEnabledInThisFile = asgn.lhsExpressions().stream()
        .flatMap(exprList -> exprList.expressions().stream())
        .findFirst()
        .flatMap(tryCast(Name.class))
        .flatMap(app -> Optional.of(app)
          .map(Name::symbol)
          .map(Symbol::usages)
          .flatMap(usages -> usages.stream().filter(CsrfDisabledCheck::isWithinCsrfEnablingStatement).findFirst()))
        .isPresent();
      if (!isCsrfEnabledInThisFile) {
        subscriptionContext.addIssue(asgn.assignedValue(), CSRFPROTECT_MISSING_MESSAGE);
      }
    }
  }

  /** Checks that an expression is some kind of <code>Flask(...)</code> constructor invocation. */
  private static boolean isFlaskAppInstantiation(Expression expr) {
    return tryCast(CallExpression.class, expr)
      .map(CallExpression::calleeSymbol)
      .filter(symb -> "flask.Flask".equals(symb.fullyQualifiedName()))
      .isPresent();
  }

  /** Detects usages like <code>CSRFProtect(a)</code>. */
  private static boolean isWithinCsrfEnablingStatement(Usage u) {
    Tree t = u.tree();
    return isWithinCall("flask_wtf.csrf.CSRFProtect", t) ||
      isWithinCall("flask_wtf.csrf.CSRFProtect.init_app", t);
  }

  /** Checks that the surroundings of <code>t</code> look like <code>expectedCalleeFqn(t)</code> */
  private static boolean isWithinCall(String expectedCalleeFqn, Tree t) {
    return Optional.ofNullable(t.parent()) // RegularArgument
      .flatMap(tryCast(RegularArgument.class))
      .map(Tree::parent) // ArgumentList
      .map(Tree::parent) // CallExpr
      .flatMap(tryCast(CallExpression.class))
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter(expectedCalleeFqn::equals)
      .isPresent();
  }
}
