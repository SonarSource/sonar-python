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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.ReachingValues;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2053")
public class PredictableSaltCheck extends PythonSubscriptionCheck {

  private static final String MISSING_SALT_MESSAGE = "Add an unpredictable salt value to this hash.";
  private static final String PREDICTABLE_SALT_MESSAGE = "Make this salt unpredictable.";
  private static final String DIFFERENT_SALT_THAN_KEY_MATERIAL_MESSAGE = "Make this salt different than the derived key material.";
  private static final String SALT_IS_USED_HERE_MESSAGE = "The salt is used in the derive method here.";
  private static final String SALT_ARGUMENT_NAME = "salt";
  private static final String PASSWORD_ARGUMENT_NAME = "password";

  private static final Map<String, ArgumentInfo> SENSITIVE_ARGUMENT_BY_FQN = Map.ofEntries(
    Map.entry("hashlib.pbkdf2_hmac", new ArgumentInfo(2, SALT_ARGUMENT_NAME, new ArgumentInfo(1, PASSWORD_ARGUMENT_NAME))),
    Map.entry("hashlib.scrypt", new ArgumentInfo(4, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("crypt.crypt", new ArgumentInfo(1, SALT_ARGUMENT_NAME)),
    Map.entry("cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC", new ArgumentInfo(2, SALT_ARGUMENT_NAME)),
    Map.entry("cryptography.hazmat.primitives.kdf.scrypt.Scrypt", new ArgumentInfo(0, SALT_ARGUMENT_NAME)),
    Map.entry("Cryptodome.Protocol.KDF.PBKDF2", new ArgumentInfo(1, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Cryptodome.Protocol.KDF.scrypt", new ArgumentInfo(1, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Cryptodome.Protocol.KDF.bcrypt", new ArgumentInfo(2, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Crypto.Protocol.KDF.PBKDF2", new ArgumentInfo(1, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Crypto.Protocol.KDF.scrypt", new ArgumentInfo(1, SALT_ARGUMENT_NAME, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))),
    Map.entry("Crypto.Protocol.KDF.bcrypt", new ArgumentInfo(2, SALT_ARGUMENT_NAME, false, new ArgumentInfo(0, PASSWORD_ARGUMENT_NAME))));

  private static final TypeMatcher DERIVE_FUNCTIONS = TypeMatchers.any(
    TypeMatchers.isType("cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC.derive"),
    TypeMatchers.isType("cryptography.hazmat.primitives.kdf.scrypt.Scrypt.derive"));

  private static final Map<String, ArgumentInfo> SALT_FUNCTION_ARGUMENTS_TO_CHECK = Map.of(
    "bytes.fromhex", new ArgumentInfo(0, "__string"),
    "bytearray.fromhex", new ArgumentInfo(0, "__string"),
    "base64.b64decode", new ArgumentInfo(0, "s"),
    "base64.b64encode", new ArgumentInfo(0, "s"),
    "base64.b32encode", new ArgumentInfo(0, "s"),
    "base64.b32decode", new ArgumentInfo(0, "s"),
    "base64.b16encode", new ArgumentInfo(0, "s"),
    "base64.b16decode", new ArgumentInfo(0, "s"));

  private static final List<ArgumentCheck> SENSITIVE_ARGUMENTS = argumentChecks(SENSITIVE_ARGUMENT_BY_FQN);
  private static final List<ArgumentCheck> SALT_FUNCTION_ARGUMENTS = argumentChecks(SALT_FUNCTION_ARGUMENTS_TO_CHECK);

  private static List<ArgumentCheck> argumentChecks(Map<String, ArgumentInfo> arguments) {
    return arguments.entrySet().stream()
      .sorted(Map.Entry.comparingByKey())
      .map(entry -> new ArgumentCheck(apiMatcher(entry.getKey()), entry.getValue()))
      .toList();
  }

  private static TypeMatcher apiMatcher(String fqn) {
    // PBKDF2HMAC is absent from older descriptors; prefer identity when it becomes available.
    if ("cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2HMAC".equals(fqn)) {
      return TypeMatchers.any(TypeMatchers.isType(fqn), TypeMatchers.withFQN(fqn));
    }
    // These libraries have no descriptors; their unresolved imports retain the public API name.
    return fqn.startsWith("Crypto.") || fqn.startsWith("Cryptodome.")
      ? TypeMatchers.withFQN(fqn)
      : TypeMatchers.isType(fqn);
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      var call = (CallExpression) ctx.syntaxNode();
      argumentInfo(SENSITIVE_ARGUMENTS, call, ctx).ifPresent(info -> checkArguments(call, info, ctx));
    });
  }

  private static Optional<ArgumentInfo> argumentInfo(List<ArgumentCheck> checks, CallExpression call, SubscriptionContext ctx) {
    return checks.stream()
      .filter(check -> check.matcher().isTrueFor(call.callee(), ctx))
      .map(ArgumentCheck::argument)
      .findFirst();
  }

  private static void checkArguments(CallExpression callExpression, ArgumentInfo argumentInfo, SubscriptionContext ctx) {
    var argument = TreeUtils.nthArgumentOrKeyword(argumentInfo.position(), argumentInfo.name(), callExpression.arguments());
    if (argument != null) {
      if (hasRaisedOnSensitiveArgument(argument, ctx)) {
        return;
      }
      if (hasRaisedOnSameArgument(argument, argumentInfo, callExpression, ctx)) {
        return;
      }
      if (hasRaisedOnSameSaltAndDerivedKeyMaterial(argument, ctx)) {
        return;
      }

    } else if (argumentInfo.required() && callExpression.arguments().stream().noneMatch(UnpackingExpression.class::isInstance)) {
      ctx.addIssue(callExpression.callee(), MISSING_SALT_MESSAGE);
    }
  }

  private static boolean hasRaisedOnSameSaltAndDerivedKeyMaterial(RegularArgument argument, SubscriptionContext ctx) {
    if (!argument.expression().is(Tree.Kind.NAME)) {
      return false;
    }
    var symbol = ((Name) argument.expression()).symbolV2();
    if (symbol == null) {
      return false;
    }
    var usagesInDeriveCall = symbol.usages().stream()
      .map(usage -> usage.tree().parent())
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .filter(regArg -> !regArg.equals(argument))
      .filter(regArg -> isUsedInDeriveCall(regArg, ctx))
      .map(RegularArgument::expression)
      .toList();

    if (usagesInDeriveCall.isEmpty()) {
      return false;
    }
    var issue = ctx.addIssue(argument, DIFFERENT_SALT_THAN_KEY_MATERIAL_MESSAGE);
    usagesInDeriveCall.forEach(arg -> issue.secondary(arg, SALT_IS_USED_HERE_MESSAGE));
    return true;
  }

  private static boolean isUsedInDeriveCall(RegularArgument regularArgument, SubscriptionContext ctx) {
    return Optional.ofNullable(regularArgument.parent())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(ArgList.class))
      .map(ArgList::parent)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::callee)
      .map(callee -> DERIVE_FUNCTIONS.isTrueFor(callee, ctx))
      .orElse(false);

  }

  private static boolean hasRaisedOnSameArgument(RegularArgument argument, ArgumentInfo argumentInfo, CallExpression callExpression, SubscriptionContext ctx) {
    return Optional.ofNullable(argumentInfo.shouldNotBeSameAsArgument())
      .map(ai -> TreeUtils.nthArgumentOrKeyword(ai.position(), ai.name(), callExpression.arguments()))
      .map(shouldNotBeSameAsArgument -> raisedOnSameArgument(argument, shouldNotBeSameAsArgument, ctx))
      .orElseGet(() -> false);

  }

  private static boolean raisedOnSameArgument(RegularArgument argument, RegularArgument shouldNotBeSameAsArgument, SubscriptionContext ctx) {
    var exp1 = argument.expression();
    var exp2 = shouldNotBeSameAsArgument.expression();
    if (exp1 instanceof Name n1
      && exp2 instanceof Name n2
      && n1.symbolV2() == n2.symbolV2()) {
      ctx.addIssue(argument, PREDICTABLE_SALT_MESSAGE).secondary(shouldNotBeSameAsArgument, "");
      return true;
    }
    return false;
  }

  private static boolean hasRaisedOnSensitiveArgument(RegularArgument regularArgument,
    SubscriptionContext ctx) {
    var secondaries = new ArrayList<Tree>();
    var expression = regularArgument.expression();
    var visited = new HashSet<Expression>();
    while (expression != null && visited.add(expression)) {
      if (expression instanceof Name name) {
        expression = getNameAssignedValueToCheck(name, secondaries, ctx);
      } else if (expression instanceof CallExpression callExpression) {
        expression = getCallExpressionArgumentValueToCheck(callExpression, ctx);
      } else if (expression instanceof StringLiteral) {
        var issue = ctx.addIssue(regularArgument, PREDICTABLE_SALT_MESSAGE);
        secondaries.forEach(t -> issue.secondary(t, ""));
        return true;
      } else {
        expression = null;
      }
    }
    return false;
  }

  @Nullable
  private static Expression getNameAssignedValueToCheck(Name name, List<Tree> secondaries, SubscriptionContext ctx) {
    var expression = ReachingValues.singleValue(name, ctx);
    if (expression == null) {
      return null;
    }
    var assignment = TreeUtils.firstAncestorOfKind(expression, Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ANNOTATED_ASSIGNMENT);
    if (assignment != null) {
      secondaries.add(assignment);
    }
    return expression;
  }

  @Nullable
  private static Expression getCallExpressionArgumentValueToCheck(CallExpression call, SubscriptionContext ctx) {
    return argumentInfo(SALT_FUNCTION_ARGUMENTS, call, ctx)
      .map(info -> TreeUtils.nthArgumentOrKeyword(info.position(), info.name(), call.arguments()))
      .map(RegularArgument::expression)
      .orElse(null);
  }

  private record ArgumentCheck(TypeMatcher matcher, ArgumentInfo argument) {
  }

  private record ArgumentInfo(int position, String name, boolean required, @Nullable ArgumentInfo shouldNotBeSameAsArgument) {
    private ArgumentInfo(int position, String name) {
      this(position, name, true, null);
    }

    private ArgumentInfo(int position, String name, ArgumentInfo shouldNotBeSameAsArgument) {
      this(position, name, true, shouldNotBeSameAsArgument);
    }
  }

}
