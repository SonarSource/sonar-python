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
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.hotspots.CommonValidationUtils.ArgumentValidator;
import org.sonar.python.checks.hotspots.CommonValidationUtils.CallValidator;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeCheckMap;

import static org.sonar.python.checks.hotspots.CommonValidationUtils.isEqualTo;
import static org.sonar.python.checks.hotspots.CommonValidationUtils.isLessThan;
import static org.sonar.python.checks.hotspots.CommonValidationUtils.isLessThanExponent;
import static org.sonar.python.semantic.SymbolUtils.qualifiedNameOrEmpty;
import static org.sonar.python.tree.TreeUtils.nthArgumentOrKeyword;
import static org.sonar.python.tree.TreeUtils.nthArgumentOrKeywordOptional;

@Rule(key = "S5344")
public class FastHashingOrPlainTextCheck extends PythonSubscriptionCheck {

  private static final String SCRYPT_PARAMETERS_MESSAGE = "Use strong scrypt parameters.";
  private static final String PBKDF2_MESSAGE = "Use at least 100 000 iterations.";
  private static final String ARGON2_MESSAGE = "Use secure Argon2 parameters.";
  private static final String BCRYPT_MESSAGE = "Use strong bcrypt parameters.";
  private static final String DJANGO_MESSAGE = "Use a secure hashing algorithm to store passwords.";

  private static final Set<String> PBKDF2_ALGOS = Set.of(
    "sha1",
    "sha256",
    "sha512"
  );
  private static final String ROUNDS = "rounds";
  private static final Set<String> DJANGO_FIRST_FORBIDDEN_HASHERS = Set.of(
    "django.contrib.auth.hashers.SHA1PasswordHasher",
    "django.contrib.auth.hashers.MD5PasswordHasher",
    "django.contrib.auth.hashers.UnsaltedSHA1PasswordHasher",
    "django.contrib.auth.hashers.UnsaltedMD5PasswordHasher",
    "django.contrib.auth.hashers.CryptPasswordHasher"
  );


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
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || isLessThanExponent(argument.expression(), 13)) {
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
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || isLessThanExponent(argument.expression(), 13)) {
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
    if (isLessThan(argument.expression(), (int) Math.pow(2, 13)) || isLessThanExponent(argument.expression(), 13)) {
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

  private static final CallValidator BCRYPT_GENSALT = new ArgumentValidator(
    0, ROUNDS, (ctx, argument) -> {
    if (isLessThan(argument.expression(), 12)) {
      ctx.addIssue(argument, BCRYPT_MESSAGE);
    }
  });
  private static final CallValidator BCRYPT_KDF = new ArgumentValidator(
    3, ROUNDS, (ctx, argument) -> {
    if (isLessThan(argument.expression(), 4096) || isLessThanExponent(argument.expression(), 12)) {
      ctx.addIssue(argument, BCRYPT_MESSAGE);
    }
  });
  private static final CallValidator PASSLIB_BCRYPT = new ArgumentValidator(
    3, ROUNDS, (ctx, argument) -> {
    if (isLessThan(argument.expression(), 12)) {
      ctx.addIssue(argument, BCRYPT_MESSAGE);
    }
  });
  private static final CallValidator FLASK_BCRYPT = new ArgumentValidator(
    1, ROUNDS, (ctx, argument) -> {
    if (isLessThan(argument.expression(), 12)) {
      ctx.addIssue(argument, BCRYPT_MESSAGE);
    }
  });


  private TypeCheckBuilder argon2IDTypeChecker = null;
  private final CallValidator argon2Type = new ArgumentValidator(
    0, "type", (ctx, argument) -> {
    if (argon2IDTypeChecker.check(argument.expression().typeV2()) == TriBool.FALSE) {
      ctx.addIssue(argument, "Use Argon2ID to improve the security of the passwords.");
    }
  });
  private TypeCheckBuilder argon2VersionTypeChecker = null;
  private final CallValidator argon2Version = new ArgumentValidator(
    1, "version", (ctx, argument) -> {
    var typeCheck = argon2VersionTypeChecker.check(argument.expression().typeV2());
    if (typeCheck == TriBool.TRUE) {
      return;
    }
    if (!isEqualTo(argument.expression(), 19)) {
      ctx.addIssue(argument, "Use the latest version of Argon2 ID.");
    }
  });

  private final Map<String, Collection<CallValidator>> callExpressionValidators = Map.ofEntries(
    Map.entry("scrypt.hash", List.of(SCRYPT_R, SCRYPT_BUFLEN, SCRYPT_N)),
    Map.entry("hashlib.scrypt", List.of(HASHLIB_R, HASHLIB_N, HASHLIB_DKLEN)),
    Map.entry("hashlib.pbkdf2_hmac", List.of(HASHLIB_PBKDF2)),
    Map.entry("cryptography.hazmat.primitives.kdf.scrypt.Scrypt", List.of(CRYPTOGRAPHY_R, CRYPTOGRAPHY_N, CRYPTOGRAPHY_LENGTH)),
    Map.entry("cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC", List.of(CRYPTOGRAPHY_PBKDF2)),
    Map.entry("passlib.handlers.scrypt.scrypt.using", List.of(PASSLIB_BLOCK_SIZE, PASSLIB_ROUNDS)),
    Map.entry("argon2.PasswordHasher", List.of(new Argon2PasswordHasherValidator(0, 1, 2))),
    Map.entry("argon2.Parameters", List.of(new Argon2PasswordHasherValidator(4, 5, 6), argon2Type, argon2Version)),
    Map.entry("argon2.low_level.hash_secret", List.of(new Argon2PasswordHasherValidator(2, 3, 4))),
    Map.entry("argon2.low_level.hash_secret_raw", List.of(new Argon2PasswordHasherValidator(2, 3, 4))),
    Map.entry("passlib.handlers.argon2._Argon2Common.using", List.of(new Argon2PasswordHasherValidator(3, 4, 5))),
    Map.entry("bcrypt.gensalt", List.of(BCRYPT_GENSALT)),
    Map.entry("bcrypt.kdf", List.of(BCRYPT_KDF)),
    Map.entry("flask_bcrypt.generate_password_hash", List.of(FLASK_BCRYPT))
  );

  private static final Map<String, Collection<CallValidator>> CALL_EXPRESSION_VALIDATORS_V1 = Map.ofEntries(
    Map.entry("flask_bcrypt.Bcrypt.generate_password_hash", List.of(FLASK_BCRYPT))
  );

  private static final Map<String, Collection<CallValidator>> QUALIFIED_EXPR_VALIDATOR = Map.of(
    "passlib.hash.pbkdf2_sha1.using", List.of(PASSLIB_PBKDF2),
    "passlib.hash.pbkdf2_sha256.using", List.of(PASSLIB_PBKDF2, PASSLIB_MISSING_ROUNDS),
    "passlib.hash.pbkdf2_sha512.using", List.of(PASSLIB_PBKDF2, PASSLIB_MISSING_ROUNDS),
    "passlib.hash.bcrypt.using", List.of(PASSLIB_BCRYPT)
  );


  private TypeCheckBuilder argon2CheapestProfileTypeChecker = null;
  private TypeCheckBuilder flaskConfigTypeChecker = null;
  private TypeCheckMap<Collection<CallValidator>> typeCheckMap = null;
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::registerTypeCheckers);
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, subscriptionContext1 -> {
      if (!"settings.py".equals(subscriptionContext1.pythonFile().fileName())) {
        return;
      }
      context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, FastHashingOrPlainTextCheck::checkDjangoHasher);
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkCallExpr);
    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, this::checkName);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, subscriptionContext -> checkAssignment(subscriptionContext, flaskConfigTypeChecker));
  }

  private static void checkDjangoHasher(SubscriptionContext subscriptionContext) {
    var stmt = (AssignmentStatement) subscriptionContext.syntaxNode();
    var lhsIsConfig = stmt.lhsExpressions().stream().findFirst()
      .map(ExpressionList::expressions)
      .flatMap(list -> list.stream().findFirst())
      .filter(expression -> expression.is(Tree.Kind.NAME))
      .filter(name -> "PASSWORD_HASHERS".equals(((Name) name).name()));

    var firstRhsString = Optional.of(stmt.assignedValue())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(ListLiteral.class))
      .map(ListLiteral::elements)
      .map(ExpressionList::expressions)
      .map(List::stream)
      .flatMap(Stream::findFirst)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .filter(stringLiteral -> DJANGO_FIRST_FORBIDDEN_HASHERS.contains(stringLiteral.trimmedQuotesValue()));

    if (lhsIsConfig.isPresent() && firstRhsString.isPresent()) {
      subscriptionContext.addIssue(firstRhsString.get(), DJANGO_MESSAGE);
    }
  }

  private void registerTypeCheckers(SubscriptionContext subscriptionContext) {
    argon2CheapestProfileTypeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("argon2.profiles.CHEAPEST");
    flaskConfigTypeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isInstanceOf("flask.config.Config");
    argon2IDTypeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("argon2.low_level.Type.ID");
    argon2VersionTypeChecker = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn("argon2.low_level.ARGON2_VERSION");
    typeCheckMap = new TypeCheckMap<>();
    callExpressionValidators.forEach((key, value) -> {
      var typeCheckBuilder = subscriptionContext.typeChecker().typeCheckBuilder().isTypeWithFqn(key);
      typeCheckMap.put(typeCheckBuilder, value);
    });
  }

  private static void checkAssignment(SubscriptionContext subscriptionContext, TypeCheckBuilder flaskConfigTypeChecker) {
    var stmt = (AssignmentStatement) subscriptionContext.syntaxNode();
    var lhsSubscription = stmt.lhsExpressions().stream().findFirst()
      .map(ExpressionList::expressions)
      .flatMap(list -> list.stream().findFirst())
      .filter(expression -> subscriptionIsFlaskBcryptConfig(expression, flaskConfigTypeChecker));

    if (lhsSubscription.isEmpty()) {
      return;
    }

    if (isLessThan(stmt.assignedValue(), 12)) {
      subscriptionContext.addIssue(stmt.assignedValue(), BCRYPT_MESSAGE);
    }
  }

  private static boolean subscriptionIsFlaskBcryptConfig(Expression expression, TypeCheckBuilder flaskConfigTypeChecker) {
    if (!expression.is(Tree.Kind.SUBSCRIPTION)) {
      return false;
    }
    var subscription = (SubscriptionExpression) expression;
    if (flaskConfigTypeChecker.check(subscription.object().typeV2()) != TriBool.TRUE) {
      return false;
    }
    var subscriptMatch = subscription.subscripts().expressions()
      .stream()
      .findFirst()
      .filter(expr -> "BCRYPT_LOG_ROUNDS".equals(CommonValidationUtils.singleAssignedString(expr)));
    return subscriptMatch.isPresent();
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

  private void checkCallExpr(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    var qualifiedName = qualifiedNameOrEmpty(callExpression);
    var configsV2 = typeCheckMap.getOptionalForType(callExpression.callee().typeV2()).orElse(List.of());
    configsV2.forEach(config -> config.validate(subscriptionContext, callExpression));
    // We need to keep some V1 checks because of SONARPY-2268.
    // When resolving members, we can get an UnknownType for the callee. This looses all type information and makes typechecking impossible
    // We could use the qualified expression mechanism to bypass this limitation, however handling it at the type inference level is better
    var configsV1 = CALL_EXPRESSION_VALIDATORS_V1.getOrDefault(qualifiedName, List.of());
    configsV1.forEach(config -> config.validate(subscriptionContext, callExpression));

    // We can't directly type check against the callee type (or FQN) for the passlib PBKDF2 because both type inference engine resolve passlib.utils.handlers.HasRounds.using or
    // passlib.ifc.PasswordHash.using which will result in false positives
    if (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
      var fqn = TreeUtils.fullyQualifiedNameFromQualifiedExpression(((QualifiedExpression) callExpression.callee())).orElse("");
      var configs = QUALIFIED_EXPR_VALIDATOR.getOrDefault(fqn, List.of());
      configs.forEach(config -> config.validate(subscriptionContext, callExpression));
    }
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
        .map(CommonValidationUtils::singleAssignedString)
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
