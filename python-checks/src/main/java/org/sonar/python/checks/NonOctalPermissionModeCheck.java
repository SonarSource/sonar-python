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

import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9160")
public class NonOctalPermissionModeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this decimal file mode with an octal literal.";
  private static final String HIGH_VALUE_MESSAGE =
    "Replace this decimal file mode with an octal literal; this decimal value exceeds 511 (0o777).";
  private static final String CONVERT_QUICK_FIX_MESSAGE = "Convert to \"%s\"";
  private static final String REPLACE_QUICK_FIX_MESSAGE = "Replace with \"%s\"";
  private static final String MODE_KEYWORD = "mode";
  private static final int MAX_UNIX_PERMISSION_MODE = 511;

  private static final Set<String> PATH_MODE_METHODS = Set.of("chmod", "lchmod", "mkdir", "touch");

  private static final TypeMatcher MODE_AT_INDEX_0_FUNCTIONS = TypeMatchers.any(
    TypeMatchers.isType("os.umask"),
    TypeMatchers.withFQN("os.umask"));

  private static final TypeMatcher MODE_AT_INDEX_1_FUNCTIONS = TypeMatchers.any(
    TypeMatchers.isType("os.chmod"),
    TypeMatchers.isType("os.fchmod"),
    TypeMatchers.withFQN("os.lchmod"),
    TypeMatchers.isType("os.mkdir"),
    TypeMatchers.isType("os.makedirs"),
    TypeMatchers.isType("os.mkfifo"),
    TypeMatchers.withFQN("os.mkfifo"),
    TypeMatchers.isType("os.mknod"),
    TypeMatchers.withFQN("os.mknod"));

  private static final TypeMatcher MODE_AT_INDEX_2_FUNCTIONS = TypeMatchers.any(
    TypeMatchers.isType("os.open"),
    TypeMatchers.withFQN("os.open"));

  private static final TypeMatcher PATH_INSTANCE = TypeMatchers.isObjectInstanceOf("pathlib.Path");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, NonOctalPermissionModeCheck::checkCall);
  }

  private static void checkCall(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();
    Optional<Integer> modeArgPosition = modeArgumentPosition(call, ctx);
    if (modeArgPosition.isEmpty()) {
      return;
    }

    RegularArgument modeArg = TreeUtils.nthArgumentOrKeyword(modeArgPosition.get(), MODE_KEYWORD, call.arguments());
    if (modeArg == null) {
      return;
    }

    Expression modeExpr = Expressions.removeParentheses(modeArg.expression());
    if (!(modeExpr instanceof NumericLiteral literal) || !isNonOctalDecimalLiteral(literal)) {
      return;
    }

    Optional<Long> numericValue = literalValueAsLong(literal);
    String message = numericValue.filter(v -> v > MAX_UNIX_PERMISSION_MODE).isPresent()
      ? HIGH_VALUE_MESSAGE
      : MESSAGE;
    var issue = ctx.addIssue(literal, message);
    if (numericValue.filter(v -> v <= MAX_UNIX_PERMISSION_MODE).isPresent()) {
      addQuickFixes(issue, literal, numericValue.get());
    }
  }

  private static void addQuickFixes(PythonCheck.PreciseIssue issue, NumericLiteral literal, long numericValue) {
    Optional<String> valueReplacement = octalValueReplacement(numericValue);
    Optional<String> intentReplacement = octalIntentReplacement(literal);

    valueReplacement.ifPresent(replacement ->
      issue.addQuickFix(PythonQuickFix.newQuickFix(String.format(CONVERT_QUICK_FIX_MESSAGE, replacement))
        .addTextEdit(TextEditUtils.replace(literal, replacement))
        .build()));

    intentReplacement
      .filter(intent -> valueReplacement.map(value -> !value.equals(intent)).orElse(true))
      .ifPresent(replacement ->
        issue.addQuickFix(PythonQuickFix.newQuickFix(String.format(REPLACE_QUICK_FIX_MESSAGE, replacement))
          .addTextEdit(TextEditUtils.replace(literal, replacement))
          .build()));
  }

  private static Optional<Integer> modeArgumentPosition(CallExpression call, SubscriptionContext ctx) {
    Expression callee = call.callee();
    if (MODE_AT_INDEX_0_FUNCTIONS.isTrueFor(callee, ctx)) {
      return Optional.of(0);
    }
    if (MODE_AT_INDEX_1_FUNCTIONS.isTrueFor(callee, ctx)) {
      return Optional.of(1);
    }
    if (MODE_AT_INDEX_2_FUNCTIONS.isTrueFor(callee, ctx)) {
      return Optional.of(2);
    }
    if (isPathModeMethod(call, ctx)) {
      return Optional.of(0);
    }
    return Optional.empty();
  }

  private static boolean isPathModeMethod(CallExpression call, SubscriptionContext ctx) {
    if (!(call.callee() instanceof QualifiedExpression qualified)) {
      return false;
    }
    if (!PATH_MODE_METHODS.contains(qualified.name().name())) {
      return false;
    }
    return PATH_INSTANCE.isTrueFor(qualified.qualifier(), ctx);
  }

  private static boolean isNonOctalDecimalLiteral(NumericLiteral literal) {
    String value = literal.valueAsString().replace("_", "");
    if ("0".equals(value)) {
      return false;
    }
    String lower = value.toLowerCase(Locale.ROOT);
    if (lower.startsWith("0o") || lower.startsWith("0x") || lower.startsWith("0b")) {
      return false;
    }
    // Python 2 octal syntax (e.g. 0755) or multi-digit zero literals
    if (value.startsWith("0")) {
      return false;
    }
    return !value.isEmpty() && value.chars().allMatch(Character::isDigit);
  }

  private static Optional<Long> literalValueAsLong(NumericLiteral literal) {
    try {
      return Optional.of(literal.valueAsLong());
    } catch (NumberFormatException e) {
      return Optional.empty();
    }
  }

  private static Optional<String> octalValueReplacement(long numericValue) {
    return Optional.of("0o" + Long.toOctalString(numericValue));
  }

  /**
   * Suggest {@code 0o...} when every digit is a valid octal digit (shell-style intent).
   * Longer values (e.g. full {@code st_mode} integers beyond four digits) are flagged without this quick fix.
   */
  private static Optional<String> octalIntentReplacement(NumericLiteral literal) {
    String original = literal.valueAsString();
    String digits = original.replace("_", "");
    if (digits.isEmpty() || digits.length() > 4) {
      return Optional.empty();
    }
    if (digits.chars().anyMatch(c -> c < '0' || c > '7')) {
      return Optional.empty();
    }
    return Optional.of("0o" + original);
  }
}
