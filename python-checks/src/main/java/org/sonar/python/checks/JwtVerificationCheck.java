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

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.InExpression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5659")
public class JwtVerificationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Don't use a JWT token without verifying its signature.";

  // https://github.com/davedoesdev/python-jwt
  // "From version 2.0.1 the namespace has changed from jwt to python_jwt, in order to avoid conflict with PyJWT"
  private static final Set<String> PROCESS_JWT_FQNS = Set.of(
    "python_jwt.process_jwt",
    "jwt.process_jwt");

  private static final Set<String> VERIFY_JWT_FQNS = Set.of(
    "python_jwt.verify_jwt",
    "jwt.verify_jwt");

  private static final Set<String> ALLOWED_KEYS_ACCESS = Set.of("jku", "jwk", "kid", "x5u", "x5c", "x5t", "xt#256");

  private static final Set<String> WHERE_VERIFY_KWARG_SHOULD_BE_TRUE_FQNS = Set.of(
    "jwt.decode",
    "jose.jws.verify");

  private static final Set<String> UNVERIFIED_FQNS = Set.of(
    "jwt.get_unverified_header",
    "jose.jwt.get_unverified_header",
    "jose.jwt.get_unverified_headers",
    "jose.jws.get_unverified_header",
    "jose.jws.get_unverified_headers",
    "jose.jwt.get_unverified_claims",
    "jose.jws.get_unverified_claims");

  private static final String VERIFY_SIGNATURE_KEYWORD = "verify_signature";

  public static final Set<String> VERIFY_SIGNATURE_OPTION_SUPPORTING_FUNCTION_FQNS = Set.of("jose.jwt.decode", "jwt.decode");

  private static final Set<String> EQUALITY_COMPARATORS = Set.of("==", "!=");

  private static final String ALGORITHMS_KEYWORD = "algorithms";

  private static final Set<String> ISSUER_CLAIM_KEY = Set.of("iss");

  private static final TypeMatcher IS_ENUM_MATCHER = TypeMatchers.isOrExtendsType("enum.Enum");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, JwtVerificationCheck::verifyCallExpression);
  }

  private static void verifyCallExpression(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();

    Symbol calleeSymbol = call.calleeSymbol();
    if (calleeSymbol == null || calleeSymbol.fullyQualifiedName() == null) {
      return;
    }

    String calleeFqn = calleeSymbol.fullyQualifiedName();
    if (WHERE_VERIFY_KWARG_SHOULD_BE_TRUE_FQNS.contains(calleeFqn)) {
      RegularArgument verifyArg = TreeUtils.argumentByKeyword("verify", call.arguments());
      if (verifyArg != null && Expressions.isFalsy(verifyArg.expression()) && !isVerifiedElsewhere(call) && !isIssuerRoutingPattern(call, ctx)) {
        ctx.addIssue(verifyArg, MESSAGE);
        return;
      }
    } else if (PROCESS_JWT_FQNS.contains(calleeFqn)) {
      Optional.ofNullable(TreeUtils.firstAncestorOfKind(call, Kind.FILE_INPUT, Kind.FUNCDEF))
        .filter(scriptOrFunction -> !TreeUtils.hasDescendant(scriptOrFunction, JwtVerificationCheck::isCallToVerifyJwt))
        .ifPresent(scriptOrFunction -> ctx.addIssue(call, MESSAGE));
    } else if (UNVERIFIED_FQNS.contains(calleeFqn) && !accessOnlyAllowedHeaderKeys(call)) {
      Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, "", call.arguments()))
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
        .map(RegularArgument::expression)
        .ifPresent(argument -> ctx.addIssue(argument, MESSAGE));
    }
    if (VERIFY_SIGNATURE_OPTION_SUPPORTING_FUNCTION_FQNS.contains(calleeFqn)) {
      Optional.ofNullable(TreeUtils.argumentByKeyword("options", call.arguments()))
        .map(RegularArgument::expression)
        .filter(JwtVerificationCheck::isListOrDictWithSensitiveEntry)
        .filter(expression -> !isVerifiedElsewhere(call) && !isIssuerRoutingPattern(call, ctx))
        .ifPresent(expression -> ctx.addIssue(expression, MESSAGE));
    }
  }

  /**
   * "Peek then verify" pattern (multi-tenant JWT key discovery): an unverified decode of a token is
   * compliant if the same token is decoded again elsewhere in the same function/module with real
   * signature verification. If the token argument can't be resolved to a symbol, we can't disprove
   * such a call exists, so we assume compliance rather than raise a false positive.
   */
  private static boolean isVerifiedElsewhere(CallExpression unverifiedCall) {
    Symbol tokenSymbol = tokenArgumentSymbol(unverifiedCall);
    if (tokenSymbol == null) {
      return true;
    }
    Tree scope = TreeUtils.firstAncestorOfKind(unverifiedCall, Kind.FILE_INPUT, Kind.FUNCDEF);
    return scope != null && TreeUtils.hasDescendant(scope, tree -> isVerifyingCallOnToken(tree, unverifiedCall, tokenSymbol));
  }

  private static boolean isVerifyingCallOnToken(Tree tree, CallExpression unverifiedCall, Symbol tokenSymbol) {
    return TreeUtils.toOptionalInstanceOf(CallExpression.class, tree)
      .filter(call -> call != unverifiedCall)
      .filter(JwtVerificationCheck::isDecodeOrVerifyCall)
      .filter(call -> tokenSymbol.equals(tokenArgumentSymbol(call)))
      .filter(call -> !isUnverifiedShape(call))
      .filter(JwtVerificationCheck::hasKeyArgument)
      .isPresent();
  }

  private static boolean hasKeyArgument(CallExpression call) {
    RegularArgument keyArg = TreeUtils.nthArgumentOrKeyword(1, "key", call.arguments());
    return keyArg != null && !Expressions.isFalsy(keyArg.expression());
  }

  private static boolean isDecodeOrVerifyCall(CallExpression call) {
    return Optional.ofNullable(call.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(fqn -> WHERE_VERIFY_KWARG_SHOULD_BE_TRUE_FQNS.contains(fqn) || VERIFY_SIGNATURE_OPTION_SUPPORTING_FUNCTION_FQNS.contains(fqn))
      .isPresent();
  }

  private static boolean isUnverifiedShape(CallExpression call) {
    RegularArgument verifyArg = TreeUtils.argumentByKeyword("verify", call.arguments());
    if (verifyArg != null && Expressions.isFalsy(verifyArg.expression())) {
      return true;
    }
    return Optional.ofNullable(TreeUtils.argumentByKeyword("options", call.arguments()))
      .map(RegularArgument::expression)
      .filter(JwtVerificationCheck::isListOrDictWithSensitiveEntry)
      .isPresent();
  }

  @Nullable
  private static Symbol tokenArgumentSymbol(CallExpression call) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, "jwt", call.arguments()))
      .map(RegularArgument::expression)
      .filter(expression -> expression.is(Kind.NAME))
      .map(expression -> ((Name) expression).symbol())
      .orElse(null);
  }

  /**
   * "Issuer routing" pattern: an unverified decode is compliant if its payload's ONLY use is reading the
   * {@code iss} claim, and that issuer is then checked against a static whitelist (enum conversion, set/list/tuple
   * membership, or an all-literal-keys dict lookup) - the real signature verification typically happens in the
   * caller, using a key resolved from the whitelisted issuer, so unlike {@link #isVerifiedElsewhere} this doesn't
   * require the same token to be re-decoded in this function. No ordering is enforced between the payload access
   * and the whitelist check (unlike {@link #isValidatedBeforeAlgorithmsUse}) - there's no sink this rule can see
   * the issuer flow into (the resolved key is typically returned or passed to a caller-supplied parameter), so
   * there's nothing to protect by requiring the whitelist check to come first.
   */
  private static boolean isIssuerRoutingPattern(CallExpression unverifiedCall, SubscriptionContext ctx) {
    Tree assignment = TreeUtils.firstAncestorOfKind(unverifiedCall, Tree.Kind.ASSIGNMENT_STMT);
    if (assignment == null) {
      return false;
    }
    List<Expression> lhsExpressions = ((AssignmentStatement) assignment).lhsExpressions().stream()
      .map(ExpressionList::expressions)
      .flatMap(Collection::stream).toList();
    if (lhsExpressions.size() != 1 || !lhsExpressions.get(0).is(Tree.Kind.NAME)) {
      return false;
    }
    Symbol payloadSymbol = ((Name) lhsExpressions.get(0)).symbol();
    if (payloadSymbol == null) {
      return false;
    }
    List<Usage> usages = getForwardUsages(payloadSymbol, unverifiedCall).toList();
    if (usages.isEmpty()) {
      return false;
    }
    Tree scope = TreeUtils.firstAncestorOfKind(unverifiedCall, Kind.FILE_INPUT, Kind.FUNCDEF);
    return scope != null && usages.stream().allMatch(usage -> isIssuerClaimAccessWhitelistedElsewhere(usage, scope, ctx));
  }

  /**
   * Whether one usage of the payload Name is a `.get("iss")`/`["iss"]` access whose extracted value (the issuer)
   * is validated against a static whitelist somewhere in {@code scope} - either via the Name it's assigned to
   * ({@link #isIssuerWhitelisted}), or, when never assigned at all, by being a direct operand of an inline
   * `in`/`not in` membership check against a literal collection ({@link #isInlineLiteralMembershipOperand}).
   * The inline case mirrors the same "no trackable symbol" limitation {@link #extractedValueSymbol} already
   * has for the algorithm-validation pattern elsewhere in this file - see review discussion on PR #1294.
   */
  private static boolean isIssuerClaimAccessWhitelistedElsewhere(Usage usage, Tree scope, SubscriptionContext ctx) {
    Tree usageParent = usage.tree().parent();
    Optional<CallExpression> getCall = getCallExprWhereDictIsAccessedWithGet(Stream.of(usageParent)).findFirst();
    if (getCall.isPresent()) {
      Stream<StringLiteral> keys = getStringLiteralKeyArgument(getCall.get());
      return isIssuerClaimKey(keys) && (isIssuerWhitelisted(getCall.get(), scope, ctx) || isInlineLiteralMembershipOperand(getCall.get()));
    }
    Optional<SubscriptionExpression> subscription = getSubscriptions(Stream.of(usageParent)).findFirst();
    if (subscription.isPresent()) {
      Stream<StringLiteral> keys = getSubscriptsStringLiteral(Stream.of(subscription.get()));
      return isIssuerClaimKey(keys) && (isIssuerWhitelisted(subscription.get(), scope, ctx) || isInlineLiteralMembershipOperand(subscription.get()));
    }
    return false;
  }

  /**
   * Whether {@code extractionSite} (e.g. {@code payload.get("iss")}) is itself a direct operand of an
   * `in`/`not in` check against a literal string collection, e.g. {@code payload.get("iss") not in {"a", "b"}}.
   * Unlike {@link #isLiteralMembershipGuardOnSymbol}, this needs no assignment/symbol - the extraction site is
   * checked directly, covering the case where the claim is compared without ever being bound to a variable.
   */
  private static boolean isInlineLiteralMembershipOperand(Tree extractionSite) {
    return TreeUtils.toOptionalInstanceOf(InExpression.class, extractionSite.parent())
      .filter(inExpr -> inExpr.leftOperand() == extractionSite)
      .map(InExpression::rightOperand)
      .flatMap(Expressions::ifNameGetSingleAssignedNonNameValue)
      .map(Expressions::expressionsFromListOrTuple)
      .filter(elements -> !elements.isEmpty() && elements.stream().allMatch(JwtVerificationCheck::isNonInterpolatedStringLiteral))
      .isPresent();
  }

  private static boolean isNonInterpolatedStringLiteral(Expression element) {
    return TreeUtils.toOptionalInstanceOf(StringLiteral.class, element)
      .filter(literal -> literal.stringElements().stream().noneMatch(StringElement::isInterpolated))
      .isPresent();
  }

  private static boolean isIssuerClaimKey(Stream<StringLiteral> keyLiterals) {
    List<StringLiteral> keys = keyLiterals.toList();
    return !keys.isEmpty() && keys.stream().allMatch(str -> ISSUER_CLAIM_KEY.contains(str.trimmedQuotesValue()));
  }

  private static boolean isIssuerWhitelisted(Tree issuerExtractionSite, Tree scope, SubscriptionContext ctx) {
    Optional<Symbol> issuerSymbol = extractedValueSymbol(issuerExtractionSite);
    if (issuerSymbol.isEmpty()) {
      return false;
    }
    Symbol symbol = issuerSymbol.get();
    return TreeUtils.hasDescendant(scope, tree -> isEnumConversionOfSymbol(tree, symbol, ctx))
      || TreeUtils.hasDescendant(scope, tree -> isLiteralMembershipGuardOnSymbol(tree, symbol))
      || TreeUtils.hasDescendant(scope, tree -> isDictLookupOnSymbolWithLiteralKeys(tree, symbol));
  }

  /**
   * Whether {@code tree} is `symbol in [...]`/`symbol not in [...]` against a list/tuple/set literal (or a Name
   * bound to one) whose elements are ALL string literals. Stricter than {@link #isAllowlistGuardOnSymbol} - that
   * one only requires the guard to exist (enough to prevent algorithm confusion, since any real check beats none),
   * but the ticket's "static whitelist" requirement here means a guard containing even one dynamic element
   * (`issuer in {"a", dynamic_value}`) doesn't qualify: the whitelist isn't actually static.
   */
  private static boolean isLiteralMembershipGuardOnSymbol(Tree tree, Symbol symbol) {
    return TreeUtils.toOptionalInstanceOf(InExpression.class, tree)
      .filter(inExpr -> isNameOfSymbol(inExpr.leftOperand(), symbol))
      .map(InExpression::rightOperand)
      .flatMap(Expressions::ifNameGetSingleAssignedNonNameValue)
      .map(Expressions::expressionsFromListOrTuple)
      .filter(elements -> !elements.isEmpty() && elements.stream().allMatch(JwtVerificationCheck::isNonInterpolatedStringLiteral))
      .isPresent();
  }

  /** Whether {@code tree} is a call passing {@code symbol} to a callee whose type is a declared Enum subclass, e.g. {@code AuthIssuer(issuer)}. */
  private static boolean isEnumConversionOfSymbol(Tree tree, Symbol symbol, SubscriptionContext ctx) {
    return TreeUtils.toOptionalInstanceOf(CallExpression.class, tree)
      .filter(call -> call.arguments().stream()
        .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
        .map(RegularArgument::expression)
        .anyMatch(expression -> isNameOfSymbol(expression, symbol)))
      .map(CallExpression::callee)
      .filter(callee -> IS_ENUM_MATCHER.isTrueFor(callee, ctx))
      .isPresent();
  }

  /** Whether {@code tree} is `symbol[key]` (or `key = symbol[...]`'s RHS) where the subscripted base is a dict literal with only string-literal keys. */
  private static boolean isDictLookupOnSymbolWithLiteralKeys(Tree tree, Symbol symbol) {
    return TreeUtils.toOptionalInstanceOf(SubscriptionExpression.class, tree)
      .filter(subscription -> subscription.subscripts().expressions().stream().anyMatch(expression -> isNameOfSymbol(expression, symbol)))
      .map(SubscriptionExpression::object)
      .flatMap(Expressions::ifNameGetSingleAssignedNonNameValue)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(DictionaryLiteral.class))
      .filter(JwtVerificationCheck::hasOnlyStringLiteralKeys)
      .isPresent();
  }

  private static boolean hasOnlyStringLiteralKeys(DictionaryLiteral dictionaryLiteral) {
    if (dictionaryLiteral.elements().isEmpty()) {
      return false;
    }
    List<KeyValuePair> pairs = dictionaryLiteral.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .toList();
    return pairs.size() == dictionaryLiteral.elements().size()
      && pairs.stream().allMatch(pair -> pair.key().is(Kind.STRING_LITERAL));
  }

  private static boolean isListOrDictWithSensitiveEntry(@Nullable Expression expression) {
    if (expression == null) {
      return false;
    } else if (expression.is(Tree.Kind.NAME)) {
      return isListOrDictWithSensitiveEntry(Expressions.singleAssignedNonNameValue((Name) expression).orElse(null));
    } else if (expression.is(Kind.DICTIONARY_LITERAL)) {
      return hasTrueVerifySignatureEntry((DictionaryLiteral) expression);
    } else if (expression.is(Kind.LIST_LITERAL)) {
      return hasTrueVerifySignatureEntry((ListLiteral) expression);
    } else if (expression.is(Kind.CALL_EXPR)) {
      return isCallToDict((CallExpression) expression)
        && hasIllegalDictKWArgument((CallExpression) expression);
    }
    return false;
  }

  private static boolean hasIllegalDictKWArgument(CallExpression expression) {
    return Optional.of(expression)
      .map(CallExpression::arguments)
      .map(arguments -> TreeUtils.argumentByKeyword(VERIFY_SIGNATURE_KEYWORD, arguments))
      .map((RegularArgument::expression))
      .filter(Expressions::isFalsy)
      .isPresent();
  }

  private static boolean isCallToDict(CallExpression expression) {
    return Optional.of(expression)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter("dict"::equals).isPresent();
  }

  private static boolean hasTrueVerifySignatureEntry(DictionaryLiteral dictionaryLiteral) {
    return dictionaryLiteral.elements().stream()
      .filter(KeyValuePair.class::isInstance)
      .map(KeyValuePair.class::cast)
      .filter(keyValuePair -> isSensitiveKey(keyValuePair.key()))
      .map(KeyValuePair::value)
      .anyMatch(Expressions::isFalsy);
  }

  private static boolean hasTrueVerifySignatureEntry(ListLiteral listLiteral) {
    return listLiteral.elements().expressions().stream()
      .filter(Tuple.class::isInstance)
      .map(Tuple.class::cast)
      .map(Tuple::elements)
      .filter(list -> list.size() == 2)
      .filter(list -> isSensitiveKey(list.get(0)))
      .map(list -> list.get(1))
      .anyMatch(Expressions::isFalsy);
  }

  private static boolean isSensitiveKey(Expression key) {
    return key.is(Kind.STRING_LITERAL) && VERIFY_SIGNATURE_KEYWORD.equals(((StringLiteral) key).trimmedQuotesValue());
  }

  private static boolean isCallToVerifyJwt(Tree t) {
    return TreeUtils.toOptionalInstanceOf(CallExpression.class, t)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter(VERIFY_JWT_FQNS::contains)
      .isPresent();
  }

  /**
   * Fail-open: a call is compliant unless a concrete unsafe usage of the header/claims value (or a value
   * extracted from it) is found. A usage shape this check doesn't recognize is not, by itself, grounds to raise -
   * only known-unsafe sinks (bare pass-through, return, disallowed key access) do, unless that same usage also
   * matches one of the safe patterns (allowed key, comparison-only, or validated-then-used-as-algorithm).
   */
  private static boolean accessOnlyAllowedHeaderKeys(CallExpression call) {
    // the call's own result can itself be a comparison operand, e.g. `jwt.get_unverified_header(token) == expected`
    // - here `call` is the operand, not `call.parent()` (which is already the COMPARISON node), unlike the
    // `.get(...)`/`[...]` chained-access shapes below where the extra level of nesting means the parent is
    // the right thing to inspect.
    if (isComparisonOperand(call)) {
      return true;
    }
    // direct chained access on the call's own result, e.g. `jwt.get_unverified_header(token).get("x5u")`,
    // is always checked - independent of whether the whole chained expression is itself assigned to some
    // unrelated name (`x5u = jwt.get_unverified_header(token).get("x5u")`).
    if (isSafeUsageSite(call.parent())) {
      return true;
    }
    Tree assignment = TreeUtils.firstAncestorOfKind(call, Tree.Kind.ASSIGNMENT_STMT);
    if (assignment == null) {
      return false;
    }
    List<Expression> lhsExpressions = ((AssignmentStatement) assignment).lhsExpressions().stream()
      .map(ExpressionList::expressions)
      .flatMap(Collection::stream).toList();
    if (lhsExpressions.size() == 1 && lhsExpressions.get(0).is(Tree.Kind.NAME)) {
      Name name = (Name) lhsExpressions.get(0);
      Symbol symbol = name.symbol();
      if (symbol != null) {
        Tree scope = TreeUtils.firstAncestorOfKind(call, Kind.FILE_INPUT, Kind.FUNCDEF);
        List<Usage> usages = getForwardUsages(symbol, call).toList();
        return !usages.isEmpty() && usages.stream().allMatch(usage -> isSafeUsage(usage, scope));
      }
    }
    return false;
  }

  /**
   * Classifies one forward usage of the header/claims Name (e.g. the `header` in `header = jwt.get_unverified_header(token)`).
   * Two things can make a usage safe, checked at two different tree depths:
   *  - the bare Name itself is compared (`header == expected`) - checked directly on {@code usage.tree()};
   *  - a value extracted from it via `.get(key)`/`[key]` is either an allowed key, compared, or validated-then-used
   *    as an algorithm - checked on {@code usageParent}, which is the `.get(...)` call or `[...]` subscription
   *    sitting immediately on top of this usage.
   * Anything else (bare pass-through, `return header`, disallowed key) falls through to `false` - this is the
   * pre-existing sink detection carried over unchanged from before this ticket, not a new way to raise issues.
   */
  private static boolean isSafeUsage(Usage usage, @Nullable Tree scope) {
    Tree usageParent = usage.tree().parent();
    if (isComparisonOperand(usage.tree())) {
      return true;
    }
    Optional<CallExpression> getCall = getCallExprWhereDictIsAccessedWithGet(Stream.of(usageParent)).findFirst();
    if (getCall.isPresent()) {
      Stream<StringLiteral> keys = getStringLiteralKeyArgument(getCall.get());
      return isSafeExtractedValueSite(getCall.get(), keys, scope);
    }
    Optional<SubscriptionExpression> subscription = getSubscriptions(Stream.of(usageParent)).findFirst();
    if (subscription.isPresent()) {
      Stream<StringLiteral> keys = getSubscriptsStringLiteral(Stream.of(subscription.get()));
      return isSafeExtractedValueSite(subscription.get(), keys, scope);
    }
    return false;
  }

  /**
   * Whether a value extracted from the header (e.g. the `.get("alg")` call itself, or a `["alg"]` subscription)
   * is safe to use unrestricted: it's directly compared, its key is in the {@link #ALLOWED_KEYS_ACCESS} allowlist,
   * or (algorithm pattern only) it's validated against an allowlist before being used as `algorithms=`.
   */
  private static boolean isSafeExtractedValueSite(Tree extractionSite, Stream<StringLiteral> keyLiterals, @Nullable Tree scope) {
    if (isComparisonOperand(extractionSite)) {
      return true;
    }
    List<StringLiteral> keys = keyLiterals.toList();
    if (!keys.isEmpty() && keys.stream().allMatch(str -> ALLOWED_KEYS_ACCESS.contains(str.trimmedQuotesValue()))) {
      return true;
    }
    return scope != null && isValidatedBeforeAlgorithmsUse(extractionSite, scope);
  }

  /** Whether {@code tree}'s parent is a `==`/`!=` comparison with {@code tree} as one of its two operands. */
  private static boolean isComparisonOperand(Tree extractionSiteOrName) {
    Tree parent = extractionSiteOrName.parent();
    if (!parent.is(Kind.COMPARISON)) {
      return false;
    }
    return EQUALITY_COMPARATORS.contains(((BinaryExpression) parent).operator().value());
  }

  /**
   * Whether {@code extractionSite} (e.g. {@code header.get("alg")}) flows, via the Name it's assigned to, into an
   * {@code in}/{@code not in} check against a literal list/tuple allowlist that appears strictly before the
   * value is passed as the {@code algorithms=} argument to a decode/verify call, anywhere in {@code scope}.
   * Line position is used as a lightweight ordering proxy rather than full control-flow dominance, matching
   * this file's existing line-number heuristics (e.g. {@link #getForwardUsages}) - the guard must be found
   * on an earlier line than the algorithms= use, otherwise the algorithm-confusion attack this rule targets
   * (using an unvalidated alg to decode, then validating too late or in unreachable code) would go undetected.
   */
  private static boolean isValidatedBeforeAlgorithmsUse(Tree extractionSite, Tree scope) {
    Optional<Symbol> symbol = extractedValueSymbol(extractionSite);
    if (symbol.isEmpty()) {
      return false;
    }
    Optional<Integer> guardLine = firstDescendantLine(scope, tree -> isAllowlistGuardOnSymbol(tree, symbol.get()));
    Optional<Integer> algorithmsUseLine = firstDescendantLine(scope, tree -> isAlgorithmsArgumentUsingSymbol(tree, symbol.get()));
    return guardLine.isPresent() && algorithmsUseLine.isPresent() && guardLine.get() < algorithmsUseLine.get();
  }

  private static Optional<Integer> firstDescendantLine(Tree tree, Predicate<Tree> predicate) {
    for (Tree child : tree.children()) {
      if (predicate.test(child)) {
        return Optional.of(child.firstToken().line());
      }
      Optional<Integer> nested = firstDescendantLine(child, predicate);
      if (nested.isPresent()) {
        return nested;
      }
    }
    return Optional.empty();
  }

  /**
   * The symbol a `.get(key)`/`[key]` extraction result is bound to, e.g. `alg` in `alg = header.get("alg")`.
   * Only handles the direct single-Name-LHS shape; if the extraction is used inline (e.g.
   * `algorithms=[header.get("alg")]`) or the target isn't a plain Name, there's no symbol to track the
   * value's later validation/use through, so this - and therefore the algorithm-validation pattern - doesn't
   * apply. That's intentional: without a trackable symbol we can't confirm the value was validated, so the
   * usage falls through to the pre-existing sink detection in {@link #isSafeUsage} instead of being exempted.
   */
  private static Optional<Symbol> extractedValueSymbol(Tree extractionSite) {
    Tree assignment = TreeUtils.firstAncestorOfKind(extractionSite, Kind.ASSIGNMENT_STMT);
    return Optional.ofNullable(assignment)
      .map(AssignmentStatement.class::cast)
      .map(AssignmentStatement::lhsExpressions)
      .filter(list -> list.size() == 1)
      .map(list -> list.get(0).expressions())
      .filter(list -> list.size() == 1 && list.get(0).is(Kind.NAME))
      .map(list -> ((Name) list.get(0)).symbol());
  }

  /** Whether {@code tree} is `symbol in [...]`/`symbol not in [...]` against a literal list/tuple (or a Name bound to one). */
  private static boolean isAllowlistGuardOnSymbol(Tree tree, Symbol symbol) {
    return TreeUtils.toOptionalInstanceOf(InExpression.class, tree)
      .filter(inExpr -> isNameOfSymbol(inExpr.leftOperand(), symbol))
      .map(InExpression::rightOperand)
      .flatMap(Expressions::ifNameGetSingleAssignedNonNameValue)
      .map(Expressions::expressionsFromListOrTuple)
      .filter(elements -> !elements.isEmpty())
      .isPresent();
  }

  /** Whether {@code tree} is a `jwt.decode`/`jose.jwt.decode` call whose `algorithms=` argument uses {@code symbol}. */
  private static boolean isAlgorithmsArgumentUsingSymbol(Tree tree, Symbol symbol) {
    return TreeUtils.toOptionalInstanceOf(CallExpression.class, tree)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter(VERIFY_SIGNATURE_OPTION_SUPPORTING_FUNCTION_FQNS::contains)
      .map(fqn -> ((CallExpression) tree).arguments())
      .map(arguments -> TreeUtils.argumentByKeyword(ALGORITHMS_KEYWORD, arguments))
      .map(RegularArgument::expression)
      .filter(algorithms -> algorithmsExpressionUsesSymbol(algorithms, symbol))
      .isPresent();
  }

  private static boolean algorithmsExpressionUsesSymbol(Expression algorithms, Symbol symbol) {
    if (isNameOfSymbol(algorithms, symbol)) {
      return true;
    }
    return Expressions.expressionsFromListOrTuple(algorithms).stream().anyMatch(element -> isNameOfSymbol(element, symbol));
  }

  private static boolean isNameOfSymbol(Expression expression, Symbol symbol) {
    return expression.is(Kind.NAME) && symbol.equals(((Name) expression).symbol());
  }

  /**
   * Same safety check as {@link #isSafeExtractedValueSite}, but for the "no assignment" path where the header
   * result is chain-accessed directly (`jwt.get_unverified_header(token).get("x5u")`) instead of first bound
   * to a Name. There's no extracted-value symbol to track here, so only comparison and allowed-key access
   * apply - the algorithm-validation pattern needs a Name to check the `in`/`not in` guard against, see
   * {@link #extractedValueSymbol}. {@code usageSite} is the node directly above the call - i.e. the
   * `QualifiedExpression` for `.get(...)` chains or the `SubscriptionExpression` for `[...]` - so the
   * comparison check must run on the resolved `.get(...)` call / subscription itself, not on {@code usageSite}:
   * for a `.get(...)` chain, {@code usageSite} is the intermediate `.get` qualified expression, which is never
   * itself a comparison operand (its parent is always the enclosing call).
   */
  private static boolean isSafeUsageSite(Tree usageSite) {
    Optional<CallExpression> getCall = getCallExprWhereDictIsAccessedWithGet(Stream.of(usageSite)).findFirst();
    if (getCall.isPresent()) {
      if (isComparisonOperand(getCall.get())) {
        return true;
      }
      return isAllowedKeyAccess(getStringLiteralKeyArgument(getCall.get()));
    }
    Optional<SubscriptionExpression> subscription = getSubscriptions(Stream.of(usageSite)).findFirst();
    if (subscription.isPresent()) {
      if (isComparisonOperand(subscription.get())) {
        return true;
      }
      return isAllowedKeyAccess(getSubscriptsStringLiteral(Stream.of(subscription.get())));
    }
    return false;
  }

  private static boolean isAllowedKeyAccess(Stream<StringLiteral> keyLiterals) {
    List<StringLiteral> keys = keyLiterals.toList();
    return !keys.isEmpty() && keys.stream().allMatch(str -> ALLOWED_KEYS_ACCESS.contains(str.trimmedQuotesValue()));
  }

  /**
   * All reads of {@code symbol} after {@code call}, using line number as a cheap proxy for "later" (this file
   * doesn't do real control-flow analysis). Binding usages are excluded: if the same variable name is rebound
   * later in the function (`header = jwt.get_unverified_header(token)` again), that's a fresh, independently
   * checked call, not a use of *this* call's result - counting it here would attribute a later call's usages
   * to this one and could wrongly flag (or wrongly clear) either call based on the other's usages.
   */
  private static Stream<Usage> getForwardUsages(Symbol symbol, CallExpression call) {
    return symbol.usages().stream()
      .filter(usage -> !usage.isBindingUsage())
      .filter(usage -> usage.tree().firstToken().line() > call.callee().firstToken().line());
  }

  private static Stream<CallExpression> getCallExprWhereDictIsAccessedWithGet(Stream<Tree> parentQualifiedExpr) {
    return parentQualifiedExpr
      .filter(parent -> parent.is(Tree.Kind.QUALIFIED_EXPR))
      .flatMap(TreeUtils.toStreamInstanceOfMapper(QualifiedExpression.class))
      .filter(expr -> "get".equals(expr.name().name()))
      .filter(expr -> expr.parent().is(Kind.CALL_EXPR))
      .map(QualifiedExpression::parent)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(CallExpression.class));
  }

  /**
   * The `.get(key)` call's key argument, as a StringLiteral if it is one. Only the first positional/keyword
   * argument is considered - `.get(key, default)`'s second argument is a fallback value, not part of the key,
   * and must not be treated as an additional accessed key (e.g. `header.get("kid", "fallback")` must still be
   * recognized as accessing only `"kid"`, not `{"kid", "fallback"}`).
   */
  private static Stream<StringLiteral> getStringLiteralKeyArgument(CallExpression getCall) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, "key", getCall.arguments()))
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .stream();
  }

  private static Stream<SubscriptionExpression> getSubscriptions(Stream<Tree> subscriptions) {
    return subscriptions
      .filter(subscription -> subscription.is(Tree.Kind.SUBSCRIPTION))
      .flatMap(TreeUtils.toStreamInstanceOfMapper(SubscriptionExpression.class));
  }

  private static Stream<StringLiteral> getSubscriptsStringLiteral(Stream<SubscriptionExpression> subscriptions) {
    return subscriptions.map(SubscriptionExpression::subscripts)
      .map(ExpressionList::expressions)
      .flatMap(Collection::stream)
      .flatMap(TreeUtils.toStreamInstanceOfMapper(StringLiteral.class));
  }

}
