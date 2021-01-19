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
package org.sonar.python.checks.hotspots;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.AbstractCallExpressionCheck;
import org.sonar.python.checks.Expressions;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.LIST_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.STRING_LITERAL;
import static org.sonar.python.checks.Expressions.singleAssignedValue;

@Rule(key = HashingDataCheck.CHECK_KEY)
public class HashingDataCheck extends AbstractCallExpressionCheck {
  public static final String CHECK_KEY = "S4790";
  private static final String MESSAGE = "Make sure that hashing data is safe here.";
  private static final Set<String> questionableFunctions = immutableSet(
    "hashlib.new",
    "cryptography.hazmat.primitives.hashes.SHA1",
    "cryptography.hazmat.primitives.hashes.MD5",
    "django.contrib.auth.hashers.make_password",
    "werkzeug.security.generate_password_hash",
    // https://github.com/Legrandin/pycryptodome
    "Cryptodome.Hash.MD2.new",
    "Cryptodome.Hash.MD4.new",
    "Cryptodome.Hash.MD5.new",
    "Cryptodome.Hash.SHA.new",
    "Cryptodome.Hash.SHA224.new",
    // https://github.com/dlitz/pycrypto
    "Crypto.Hash.MD2.new",
    "Crypto.Hash.MD4.new",
    "Crypto.Hash.MD5.new",
    "Crypto.Hash.SHA.new",
    "Crypto.Hash.SHA224.new"
    );
  private static final Set<String> questionableHashlibAlgorithm = immutableSet(
    "hashlib.md5",
    "hashlib.sha224"
  );

  private static final Set<String> questionablePasslibAlgorithm = Stream.of(
    "apr_md5_crypt", "bigcrypt", "bsd_nthash", "bsdi_crypt",
    "cisco_asa", "cisco_pix", "cisco_type7", "crypt16",
    "des_crypt", "django_des_crypt", "django_salted_md5",
    "django_salted_sha1", "dlitz_pbkdf2_sha1",
    "hex_md4", "hex_md5", "hex_sha1", "ldap_bsdi_crypt", "ldap_des_crypt", "ldap_hex_md5",
    "ldap_plaintext", "ldap_salted_md5",
    "ldap_salted_sha1", "ldap_sha1", "ldap_sha1_crypt", "lmhash", "md5_crypt", "mssql2000",
    "mssql2005", "mysql323", "mysql41", "nthash", "oracle10", "plaintext",
    "postgres_md5", "roundup_plaintext", "sha1_crypt", "sun_md5_crypt")
    .map(hasher -> "passlib.hash." + hasher)
    .collect(Collectors.toSet());


  private static final Set<String> questionableDjangoHashers = Stream.of(
    "SHA1PasswordHasher", "MD5PasswordHasher", "UnsaltedSHA1PasswordHasher",
    "UnsaltedMD5PasswordHasher", "CryptPasswordHasher")
    .map(hasher -> "django.contrib.auth.hashers." + hasher)
    .collect(Collectors.toSet());

  private static final Set<String> questionableHasher = immutableSet("crypt", "unsalted_sha1", "unsalted_md5", "sha1", "md5");

  @Override
  public void initialize(Context context) {
    super.initialize(context);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, HashingDataCheck::checkOverwriteDjangoHashers);
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, HashingDataCheck::checkCreatingCustomHasher);
    context.registerSyntaxNodeConsumer(NAME, HashingDataCheck::checkQuestionableHashingAlgorithm);
  }

  @Override
  protected boolean isException(CallExpression callExpression) {
    return isSafeDjangoMakePasswordFunction(callExpression) || isSafeWerkzeugHashFunction(callExpression);
  }

  private static boolean isSafeDjangoMakePasswordFunction(CallExpression callExpression) {
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol != null
      && "django.contrib.auth.hashers.make_password".equals(calleeSymbol.fullyQualifiedName())) {
      RegularArgument hasher = TreeUtils.nthArgumentOrKeyword(2, "hasher", callExpression.arguments());
      if (hasher == null) {
        return true;
      }
      StringLiteral value = (StringLiteral) getValue(hasher.expression(), STRING_LITERAL);
      return value == null || !questionableHasher.contains(value.trimmedQuotesValue());
    }
    return false;
  }

  private static boolean isSafeWerkzeugHashFunction(CallExpression callExpression) {
    Symbol calleeSymbol = callExpression.calleeSymbol();
    if (calleeSymbol != null
      && "werkzeug.security.generate_password_hash".equals(calleeSymbol.fullyQualifiedName())) {
      RegularArgument method = TreeUtils.nthArgumentOrKeyword(1, "method", callExpression.arguments());
      if (method == null) {
        return true;
      }
      StringLiteral value = (StringLiteral) getValue(method.expression(), STRING_LITERAL);
      if (value == null) {
        return true;
      }
      String val = value.trimmedQuotesValue();
      return !val.equals("md5") && !val.equals("sha224");
    }
    return false;
  }

  private static void checkOverwriteDjangoHashers(SubscriptionContext ctx) {
    AssignmentStatement assignmentStatementTree = (AssignmentStatement) ctx.syntaxNode();

    if (isOverwritingDjangoHashers(assignmentStatementTree.lhsExpressions())) {
      List<Expression> weakHashers = getWeakHashers(assignmentStatementTree.assignedValue());
      if (!weakHashers.isEmpty()) {
        PreciseIssue preciseIssue = ctx.addIssue(assignmentStatementTree, MESSAGE);
        weakHashers.forEach(expression -> preciseIssue.secondary(expression, null));
      }
      return;
    }

    // checks for `PASSWORD_HASHERS = []` in a global_settings.py file
    if (ctx.pythonFile().fileName().equals("global_settings.py") &&
      assignmentStatementTree.lhsExpressions().stream()
        .flatMap(pelt -> pelt.expressions().stream())
        .anyMatch(expression -> expression.firstToken().value().equals("PASSWORD_HASHERS"))) {
      List<Expression> weakHashers = getWeakHashers(assignmentStatementTree.assignedValue());
      if (!weakHashers.isEmpty()) {
        PreciseIssue preciseIssue = ctx.addIssue(assignmentStatementTree, MESSAGE);
        weakHashers.forEach(expression -> preciseIssue.secondary(expression, null));
      }
    }
  }

  private static List<Expression> getWeakHashers(Expression assignedValue) {
    ListLiteral listLiteral = (ListLiteral) getValue(assignedValue, LIST_LITERAL);
    if (listLiteral != null) {
      return listLiteral.elements().expressions().stream()
        .filter(HashingDataCheck::isWeakHasher)
        .collect(Collectors.toList());
    }
    return Collections.emptyList();
  }

  private static boolean isWeakHasher(Expression expression) {
    StringLiteral value = (StringLiteral) getValue(expression, STRING_LITERAL);
    return value != null && questionableDjangoHashers.contains(value.trimmedQuotesValue());
  }

  @CheckForNull
  private static Expression getValue(Expression expression, Tree.Kind kind) {
    Expression expr = expression;
    if (expression.is(NAME)) {
      expr = singleAssignedValue(((Name) expression));
    }
    if (expr != null && expr.is(kind)) {
      return expr;
    }
    return null;
  }

  /**
   * checks for `settings.PASSWORD_HASHERS = value`
   */
  private static boolean isOverwritingDjangoHashers(List<ExpressionList> lhsExpressions) {
    for (ExpressionList expr : lhsExpressions) {
      for (Expression expression : expr.expressions()) {
        Expression baseExpr = Expressions.removeParentheses(expression);
        if (baseExpr.is(Tree.Kind.QUALIFIED_EXPR)) {
          QualifiedExpression qualifiedExpression = (QualifiedExpression) baseExpr;
          if (qualifiedExpression.symbol() != null
            && "django.conf.settings.PASSWORD_HASHERS".equals(qualifiedExpression.symbol().fullyQualifiedName())) {
            return true;
          }
        }
      }
    }
    return false;
  }

  private static void checkQuestionableHashingAlgorithm(SubscriptionContext ctx) {
    Name name = (Name) ctx.syntaxNode();
    if (isWithinImport(name)) {
      return;
    }
    String fullyQualifiedName = name.symbol() != null ? name.symbol().fullyQualifiedName() : "";
    if (questionableHashlibAlgorithm.contains(fullyQualifiedName) || questionablePasslibAlgorithm.contains(fullyQualifiedName)) {
      ctx.addIssue(name, MESSAGE);
    }
  }

  private static String getQualifiedName(Expression node) {
    if (node instanceof HasSymbol) {
      Symbol symbol = ((HasSymbol) node).symbol();
      return symbol != null ? symbol.fullyQualifiedName() : "";
    }
    return "";
  }

  private static void checkCreatingCustomHasher(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();
    ArgList argList = classDef.args();
    if (argList != null) {
      argList.arguments()
        .stream()
        .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
        .map(RegularArgument.class::cast)
        .filter(arg -> questionableDjangoHashers.contains(getQualifiedName(arg.expression())))
        .forEach(arg -> ctx.addIssue(arg, MESSAGE));
    }
  }

  @Override
  protected Set<String> functionsToCheck() {
    return questionableFunctions;
  }

  @Override
  protected String message() {
    return MESSAGE;
  }
}
