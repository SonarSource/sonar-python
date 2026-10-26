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
package org.sonar.python.checks.hotspots;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.FunctionLike;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2245")
public class PseudoRandomCheck extends PythonSubscriptionCheck {

  private static final Set<String> FUNCTION_NAMES = Set.of(
    "random",
    "getrandbits",
    "randint",
    "sample",
    "choice",
    "choices",
    "randbytes",
    "randrange",
    "shuffle");
  private static final String RANDOM_PACKAGE_PREFIX = "random.";
  private static final String RANDOM_CLASS_PREFIX = "random.Random.";
  private static final Set<String> QUALIFIERS_TO_SKIP = Set.of("random.SystemRandom");
  private static final Set<String> FUNCTIONS_TO_CHECK = getFunctionsFullyQualifiedNames();
  private static final Set<String> CRYPTO_MODULES = Set.of(
    "cryptography", "Crypto", "Cryptodome", "OpenSSL", "nacl", "pysodium", "tink", "rsa", "ecdsa",
    "passlib", "bcrypt", "argon2", "scrypt",
    "jwt", "jose", "jwcrypto", "authlib", "oauthlib", "requests_oauthlib", "itsdangerous", "pyotp", "pyseto", "paseto",
    "hmac", "ssl", "flask_login", "flask_jwt_extended", "django.contrib.auth");
  private static final Set<String> SECURITY_KEYWORDS = Set.of(
    "aes", "asymmetric", "auth", "captcha", "certificate", "chacha20", "challenge", "cipher", "crypto", "cryptography", "decrypt", "ecc", "encrypt",
    "encryption", "hmac", "iv", "nonce", "password", "pbkdf2", "poly1305", "randombytes", "rsa", "salt", "scrypt", "secret",
    "secure", "security", "sessionid", "sid", "signature", "token", "verify");
  private static final Pattern DIGITS = Pattern.compile("\\d+");
  private static final Pattern WORD_SPLIT = Pattern.compile("_+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])");
  public static final String MESSAGE = "Make sure that using this pseudorandom number generator is safe here.";

  private boolean hasCryptoImport;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      CryptoImportFinder finder = new CryptoImportFinder();
      ctx.syntaxNode().accept(finder);
      hasCryptoImport = finder.found;
      localCache.clear();
      headerCache.clear();
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      if (skip(callExpression) || !(hasCryptoImport || hasSecurityKeywordInScope(callExpression) || isSecurityCookieValue(callExpression))) {
        return;
      }

      Symbol symbol = callExpression.calleeSymbol();
      Optional.ofNullable(symbol)
        .map(Symbol::fullyQualifiedName)
        .filter(FUNCTIONS_TO_CHECK::contains)
        .ifPresent(functionFqn -> ctx.addIssue(callExpression, MESSAGE));
    });
  }

  private static Set<String> getFunctionsFullyQualifiedNames() {
    return FUNCTION_NAMES.stream()
      .flatMap(functionName -> Stream.of(RANDOM_PACKAGE_PREFIX + functionName, RANDOM_CLASS_PREFIX + functionName))
      .collect(Collectors.toSet());
  }

  private static boolean skip(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(CallExpression::callee)
      .filter(QualifiedExpression.class::isInstance)
      .map(QualifiedExpression.class::cast)
      .map(QualifiedExpression::qualifier)
      .map(Expression::type)
      .filter(type -> QUALIFIERS_TO_SKIP.stream().anyMatch(type::mustBeOrExtend))
      .isPresent();
  }

  // Only the value expression gains context: a security-named cookie must not
  // make unrelated jitter, expiry arguments or other cookie values report.
  private static boolean isSecurityCookieValue(CallExpression randomCall) {
    for (Tree tree = randomCall.parent(); tree != null && !(tree instanceof Statement) && !(tree instanceof FunctionLike); tree = tree.parent()) {
      if (tree instanceof RegularArgument argument && tree.parent().parent() instanceof CallExpression call
        && argument == TreeUtils.nthArgumentOrKeyword(1, "value", call.arguments()) && isSecurityCookieCall(call)) {
        return true;
      }
    }
    return false;
  }

  private static boolean isSecurityCookieCall(CallExpression call) {
    if (!(call.callee() instanceof QualifiedExpression callee) || !CookieApi.SET_COOKIE_METHOD_NAME.equals(callee.name().name())
      || CookieApi.RESPONSE_BASE_TYPES.stream().noneMatch(callee.qualifier().type()::mustBeOrExtend)) {
      return false;
    }
    RegularArgument key = TreeUtils.nthArgumentOrKeyword(0, "key", call.arguments());
    return key != null && unwrap(key.expression()) instanceof StringLiteral literal && isSecurityName(literal.trimmedQuotesValue());
  }

  private static class CryptoImportFinder extends BaseTreeVisitor {
    boolean found = false;

    private static boolean isCryptoModule(String module) {
      return CRYPTO_MODULES.stream().anyMatch(m -> module.equals(m) || module.startsWith(m + "."));
    }

    private static String dotted(List<Name> names) {
      return names.stream().map(Name::name).collect(Collectors.joining("."));
    }

    @Override
    public void visitImportName(ImportName importName) {
      importName.modules().stream().map(AliasedName::dottedName).forEach(d -> found |= isCryptoModule(dotted(d.names())));
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      if (importFrom.module() == null || !importFrom.dottedPrefixForModule().isEmpty()) {
        return;
      }
      String module = dotted(importFrom.module().names());
      found |= isCryptoModule(module);
      // "from django.contrib import auth" is "django.contrib.auth"
      importFrom.importedNames().forEach(n -> found |= isCryptoModule(module + "." + dotted(n.dottedName().names())));
    }

    @Override
    public void visitCallExpression(CallExpression call) {
      Symbol symbol = call.calleeSymbol();
      if (symbol != null && "importlib.import_module".equals(symbol.fullyQualifiedName())
        && !call.arguments().isEmpty()
        && call.arguments().get(0) instanceof RegularArgument arg
        && arg.expression() instanceof StringLiteral literal) {
        found |= isCryptoModule(literal.trimmedQuotesValue());
      }
      super.visitCallExpression(call);
    }
  }

  /** Collects identifier names of a scope without entering nested scopes. */
  private static class NameCollector extends BaseTreeVisitor {
    final Set<String> names = new HashSet<>();

    @Override
    public void visitName(Name name) {
      names.add(name.name());
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      // nested scope
    }

    @Override
    public void visitLambda(LambdaExpression lambda) {
      // nested scope
    }

    @Override
    public void visitClassDef(ClassDef classDef) {
      // nested scope
    }

    // "verify = lambda x: x": the name belongs to the lambda, not to this scope
    @Override
    public void visitAssignmentStatement(AssignmentStatement assignment) {
      if (!(unwrap(assignment.assignedValue()) instanceof LambdaExpression)) {
        super.visitAssignmentStatement(assignment);
      }
    }

    @Override
    public void visitAnnotatedAssignment(AnnotatedAssignment assignment) {
      Expression value = assignment.assignedValue();
      if (value == null || !(unwrap(value) instanceof LambdaExpression)) {
        super.visitAnnotatedAssignment(assignment);
      }
    }
  }

  private static Expression unwrap(Expression e) {
    return e instanceof ParenthesizedExpression p ? unwrap(p.expression()) : e;
  }

  // one scan per scope: random calls are often numerous in the same scope
  private final Map<Tree, Boolean> localCache = new HashMap<>();
  private final Map<Tree, Boolean> headerCache = new HashMap<>();

  private boolean hasSecurityKeywordInScope(CallExpression call) {
    boolean innermost = true;
    for (Tree t = call.parent(); t != null; t = t.parent()) {
      if (t instanceof FunctionLike fn) {
        if ((innermost && cached(localCache, fn, f -> scan(c -> collectBody(fn, c))))
          || cached(headerCache, fn, f -> scan(c -> collectHeader(fn, c)))) {
          return true;
        }
        innermost = false;
      } else if (innermost && (t instanceof ClassDef || t.is(Tree.Kind.FILE_INPUT))) {
        if (cached(localCache, t, PseudoRandomCheck::scanScope)) {
          return true;
        }
        innermost = false;
      }
    }
    return false;
  }

  private static boolean cached(Map<Tree, Boolean> cache, Tree key, Predicate<Tree> scan) {
    return cache.computeIfAbsent(key, scan::test);
  }

  private static boolean scanScope(Tree scope) {
    Tree body = scope instanceof ClassDef classDef ? classDef.body() : scope;
    return scan(body::accept);
  }

  private static boolean scan(Consumer<NameCollector> collect) {
    NameCollector collector = new NameCollector();
    collect.accept(collector);
    return collector.names.stream().anyMatch(PseudoRandomCheck::isSecurityName);
  }

  private static void collectBody(FunctionLike fn, NameCollector collector) {
    if (fn instanceof FunctionDef def) {
      def.body().accept(collector);
    } else {
      ((LambdaExpression) fn).expression().accept(collector);
    }
  }

  /** Name, parameters and, for a lambda, the names receiving it or its result ("token = list(map(lambda: ..., xs))"). */
  private static void collectHeader(FunctionLike fn, NameCollector collector) {
    if (fn instanceof FunctionDef def) {
      collector.names.add(def.name().name());
    } else {
      Tree t = fn.parent();
      while (t != null && !(t instanceof Statement) && !(t instanceof FunctionLike)) {
        t = t.parent();
      }
      if (t instanceof AssignmentStatement a) {
        a.lhsExpressions().forEach(l -> l.accept(collector));
      } else if (t instanceof AnnotatedAssignment a) {
        a.variable().accept(collector);
      }
    }
    if (fn.parameters() != null) {
      fn.parameters().accept(collector);
    }
  }

  static boolean isSecurityName(String identifier) {
    List<String> words = Stream.of(WORD_SPLIT.split(identifier))
      .filter(w -> !w.isEmpty())
      .map(w -> w.toLowerCase(Locale.ROOT))
      .toList();
    return IntStream.range(0, words.size()).anyMatch(i -> {
      String word = words.get(i);
      // "pbkdf2" matches whole, "aes256" matches through its letters-only part
      return SECURITY_KEYWORDS.contains(word)
        || Stream.of(DIGITS.split(word)).anyMatch(SECURITY_KEYWORDS::contains)
        || (i > 0 && SECURITY_KEYWORDS.contains(words.get(i - 1) + word));
    });
  }

}
