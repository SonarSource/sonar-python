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
package org.sonar.python.checks.hotspots;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.checks.hotspots.CommonValidationUtils.ArgumentValidator;
import org.sonar.python.checks.hotspots.CommonValidationUtils.CallValidator;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;

import static org.sonar.python.checks.hotspots.CommonValidationUtils.isEqualTo;
import static org.sonar.python.checks.hotspots.CommonValidationUtils.isLessThan;
import static org.sonar.python.semantic.SymbolUtils.qualifiedNameOrEmpty;
import static org.sonar.python.tree.TreeUtils.nthArgumentOrKeyword;
import static org.sonar.python.tree.TreeUtils.nthArgumentOrKeywordOptional;

@Rule(key = "S5344")
public class FastHashingOrPlainTextCheck extends PythonSubscriptionCheck {

  private static final String SCRYPT_PARAMETERS_MESSAGE = "Use strong scrypt parameters.";
  private static final String PBKDF2_MESSAGE = "Use at least 100 000 iterations.";
  private static final String ARGON2_MESSAGE = "Use secure Argon2 parameters.";
  private static final Set<String> PBKDF2_ALGOS = Set.of(
    "sha1",
    "sha256",
    "sha512"
  );
  private static final String ROUNDS = "rounds";


  private static final ArgumentValidator SCRYPT_R = new ArgumentValidator(
    3, "r", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 8)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator SCRYPT_BUFLEN = new ArgumentValidator(
    5, "buflen", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 32)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator SCRYPT_N = new ArgumentValidator(
    2, "N", (ctx, argument) -> {
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || CommonValidationUtils.isLessThanExponent(argument.expression(), 13)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });

  private static final ArgumentValidator HASHLIB_R = new ArgumentValidator(
    3, "r", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 8)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator HASHLIB_N = new ArgumentValidator(
    2, "n", (ctx, argument) -> {
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || CommonValidationUtils.isLessThanExponent(argument.expression(), 13)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator HASHLIB_DKLEN = new ArgumentValidator(
    6, "dklen", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 32)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });

  private static final ArgumentValidator CRYPTOGRAPHY_R = new ArgumentValidator(
    3, "r", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 8)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator CRYPTOGRAPHY_N = new ArgumentValidator(
    2, "n", (ctx, argument) -> {
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || CommonValidationUtils.isLessThanExponent(argument.expression(), 13)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator CRYPTOGRAPHY_LENGTH = new ArgumentValidator(
    1, "length", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 32)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });

  private static final ArgumentValidator PASSLIB_BLOCK_SIZE = new ArgumentValidator(
    3, "block_size", (ctx, argument) -> {
    if (isLessThan(argument.expression(), 8)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });
  private static final ArgumentValidator PASSLIB_ROUNDS = new ArgumentValidator(
    2, ROUNDS, (ctx, argument) -> {
    if (isLessThan(argument.expression(), 12)) {
      ctx.addIssue(argument, SCRYPT_PARAMETERS_MESSAGE);
    }
  });


  private static final ArgumentValidator PASSLIB_PBKDF2 = new ArgumentValidator(
    2, ROUNDS, (ctx, argument) -> {
    if (isLessThan(argument.expression(), 100_000)) {
      ctx.addIssue(argument, PBKDF2_MESSAGE);
    }
  });
  private static final CallValidator CRYPTOGRAPHY_PBKDF2 = new PBKDF2Validator(0, "algorithm", 3, "iterations");
  private static final CallValidator HASHLIB_PBKDF2 = new PBKDF2Validator(0, "hash_name", 3, "iterations");
  private static final CallValidator PASSLIB_MISSING_ROUNDS = new MissingArgumentValidator(
    2, ROUNDS, PBKDF2_MESSAGE
  );


  private static final Map<String, Collection<CallValidator>> CALL_EXPRESSION_VALIDATORS = Map.ofEntries(
    Map.entry("scrypt.hash", List.of(SCRYPT_R, SCRYPT_BUFLEN, SCRYPT_N)),
    Map.entry("hashlib.scrypt", List.of(HASHLIB_R, HASHLIB_N, HASHLIB_DKLEN)),
    Map.entry("hashlib.pbkdf2_hmac", List.of(HASHLIB_PBKDF2)),
    Map.entry("cryptography.hazmat.primitives.kdf.scrypt.Scrypt", List.of(CRYPTOGRAPHY_R, CRYPTOGRAPHY_N, CRYPTOGRAPHY_LENGTH)),
    Map.entry("cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC", List.of(CRYPTOGRAPHY_PBKDF2)),
    Map.entry("passlib.hash.scrypt.using", List.of(PASSLIB_BLOCK_SIZE, PASSLIB_ROUNDS)),
    Map.entry("argon2.PasswordHasher", List.of(new Argon2PasswordHasherValidator(0, 1, 2))),
    Map.entry("argon2.Parameters", List.of(new Argon2PasswordHasherValidator(4, 5, 6))),
    Map.entry("argon2.low_level.hash_secret", List.of(new Argon2PasswordHasherValidator(2, 3, 4))),
    Map.entry("argon2.low_level.hash_secret_raw", List.of(new Argon2PasswordHasherValidator(2, 3, 4))),
    Map.entry("passlib.handlers.argon2._Argon2Common.using", List.of(new Argon2PasswordHasherValidator(3, 4, 5)))
  );

  private static final Map<String, Collection<CallValidator>> QUALIFIED_EXPR_VALIDATOR = Map.of(
    "passlib.hash.pbkdf2_sha1.using", List.of(PASSLIB_PBKDF2),
    "passlib.hash.pbkdf2_sha256.using", List.of(PASSLIB_PBKDF2, PASSLIB_MISSING_ROUNDS),
    "passlib.hash.pbkdf2_sha512.using", List.of(PASSLIB_PBKDF2, PASSLIB_MISSING_ROUNDS)
  );


  private TypeCheckBuilder argon2CheapestProfileTypeChecker = null;
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::registerTypeCheckers);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, FastHashingOrPlainTextCheck::checkCallExpr);
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, this::checkName);
  }

  private void registerTypeCheckers(SubscriptionContext subscriptionContext) {
    argon2CheapestProfileTypeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("argon2.profiles.CHEAPEST");
  }

  private void checkName(SubscriptionContext subscriptionContext) {
    var name = (Name) subscriptionContext.syntaxNode();
    if (argon2CheapestProfileTypeChecker.check(name.typeV2()) != TriBool.TRUE) {
      return;
    }
    var ancestorAssign = ((AssignmentStatement) TreeUtils.firstAncestorOfKind(name, Tree.Kind.ASSIGNMENT_STMT));
    if (ancestorAssign != null && isChildOf(ancestorAssign, name)) {
      return;
    }
    subscriptionContext.addIssue(name, "Use a secure Argon2 profile.");
  }

  private static boolean isChildOf(AssignmentStatement ancestorAssign, Name name) {
    return ancestorAssign.lhsExpressions().stream().flatMap(expressionList -> expressionList.children().stream()).anyMatch(tree -> tree == name);
  }

  private static void checkCallExpr(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    var qualifiedName = qualifiedNameOrEmpty(callExpression);
    var configs = CALL_EXPRESSION_VALIDATORS.getOrDefault(qualifiedName, List.of());

    configs.forEach(config -> config.validate(subscriptionContext, callExpression));

    // We can't directly type check against the callee type (or FQN) for the passlib PBKDF2 because both type inference engine resolve passlib.utils.handlers.HasRounds.using
    // which will result in false positives
    if (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
      var fqn = TreeUtils.fullyQualifiedNameFromQualifiedExpression(((QualifiedExpression) callExpression.callee())).orElse("");
      configs = QUALIFIED_EXPR_VALIDATOR.getOrDefault(fqn, List.of());
      configs.forEach(config -> config.validate(subscriptionContext, callExpression));
    }
  }

  private static String singleAssignedString(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return Expressions.singleAssignedNonNameValue(((Name) expression))
        .map(FastHashingOrPlainTextCheck::singleAssignedString)
        .orElse("");
    }
    return expression.is(Tree.Kind.STRING_LITERAL) ? ((StringLiteral) expression).trimmedQuotesValue() : "";
  }

  record PBKDF2Validator(
    int algoPosition,
    String algoKeyword,
    int iterationsPosition,
    String iterationsKeyword
  ) implements CallValidator {
    @Override
    public void validate(SubscriptionContext ctx, CallExpression callExpression) {
      var algoArgument =
        nthArgumentOrKeywordOptional(algoPosition, algoKeyword, callExpression.arguments());
      var algoString = algoArgument
        .map(RegularArgument::expression)
        .map(FastHashingOrPlainTextCheck::singleAssignedString)
        .orElse("");
      if (!PBKDF2_ALGOS.contains(algoString)) {
        return;
      }
      nthArgumentOrKeywordOptional(iterationsPosition, iterationsKeyword, callExpression.arguments())
        .filter(arg -> isLessThan(arg.expression(), 100_000))
        .ifPresent(arg -> ctx.addIssue(arg, PBKDF2_MESSAGE));
    }
  }
  record MissingArgumentValidator(int position, String keywordName, String message) implements CallValidator {
    @Override
    public void validate(SubscriptionContext ctx, CallExpression callExpression) {
      nthArgumentOrKeywordOptional(position, keywordName, callExpression.arguments()).ifPresentOrElse(regularArgument -> {
      }, () -> ctx.addIssue(callExpression.callee(), message));
    }
  }

  record Argon2PasswordHasherValidator(
    int timeCostPosition,
    int memoryCostPosition,
    int parallelismPosition
  ) implements CallValidator {

    @Override
    public void validate(SubscriptionContext ctx, CallExpression callExpression) {
      var timeCostArgument =
        nthArgumentOrKeyword(timeCostPosition, "time_cost", callExpression.arguments());
      var memoryCostArgument =
        nthArgumentOrKeyword(memoryCostPosition, "memory_cost", callExpression.arguments());
      var parallelismArgument =
        nthArgumentOrKeyword(parallelismPosition, "parallelism", callExpression.arguments());

      var isTimeCostNOk = timeCostArgument != null && isLessThan(timeCostArgument.expression(), 5);
      var isMemoryCostNOk = memoryCostArgument != null && isLessThan(memoryCostArgument.expression(), 7168);
      var isParallelismNOk = parallelismArgument != null && isEqualTo(parallelismArgument.expression(), 1);

      if (isMemoryCostNOk && isTimeCostNOk && isParallelismNOk) {
        ctx.addIssue(callExpression.callee(), ARGON2_MESSAGE);
      }
    }
  }
}
