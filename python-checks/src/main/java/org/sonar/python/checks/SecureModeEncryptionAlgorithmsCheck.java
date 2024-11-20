/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;

@Rule(key = "S5542")
public class SecureModeEncryptionAlgorithmsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use secure mode and padding scheme.";
  private static final Set<String> PYCA_RSA_KEY_METHODS = new HashSet<>(Arrays.asList(
    "cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey.decrypt",
    "cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey.encrypt"
  ));

  private static final Set<String> PYCA_VULNERABLE_MODES = new HashSet<>(Arrays.asList(
    "cryptography.hazmat.primitives.ciphers.modes.CBC",
    "cryptography.hazmat.primitives.ciphers.modes.ECB"
  ));

  private static final Map<String, List<String>> PYCRYPTO_VULNERABLE_MODES_BY_API = new HashMap<>();
  static {
    PYCRYPTO_VULNERABLE_MODES_BY_API.put("Crypto.Cipher.DES.new", Arrays.asList("Crypto.Cipher.DES.MODE_ECB", "Crypto.Cipher.DES.MODE_CBC"));
    PYCRYPTO_VULNERABLE_MODES_BY_API.put("Crypto.Cipher.CAST.new", Arrays.asList("Crypto.Cipher.CAST.MODE_ECB", "Crypto.Cipher.CAST.MODE_CBC"));
    PYCRYPTO_VULNERABLE_MODES_BY_API.put("Crypto.Cipher.DES3.new", Arrays.asList("Crypto.Cipher.DES3.MODE_ECB", "Crypto.Cipher.DES3.MODE_CBC"));
    PYCRYPTO_VULNERABLE_MODES_BY_API.put("Crypto.Cipher.ARC2.new", Arrays.asList("Crypto.Cipher.ARC2.MODE_ECB", "Crypto.Cipher.ARC2.MODE_CBC"));
    PYCRYPTO_VULNERABLE_MODES_BY_API.put("Crypto.Cipher.Blowfish.new", Arrays.asList("Crypto.Cipher.Blowfish.MODE_ECB", "Crypto.Cipher.Blowfish.MODE_CBC"));
    PYCRYPTO_VULNERABLE_MODES_BY_API.put("Crypto.Cipher.AES.new", Arrays.asList("Crypto.Cipher.AES.MODE_ECB", "Crypto.Cipher.AES.MODE_CBC"));
  }

  private static final Map<String, List<String>> CRYPTODOMEX_VULNERABLE_MODES_BY_API = new HashMap<>();
  static {
    CRYPTODOMEX_VULNERABLE_MODES_BY_API.put("Cryptodome.Cipher.DES.new", Arrays.asList("Cryptodome.Cipher.DES.MODE_ECB", "Cryptodome.Cipher.DES.MODE_CBC"));
    CRYPTODOMEX_VULNERABLE_MODES_BY_API.put("Cryptodome.Cipher.CAST.new", Arrays.asList("Cryptodome.Cipher.CAST.MODE_ECB", "Cryptodome.Cipher.CAST.MODE_CBC"));
    CRYPTODOMEX_VULNERABLE_MODES_BY_API.put("Cryptodome.Cipher.DES3.new", Arrays.asList("Cryptodome.Cipher.DES3.MODE_ECB", "Cryptodome.Cipher.DES3.MODE_CBC"));
    CRYPTODOMEX_VULNERABLE_MODES_BY_API.put("Cryptodome.Cipher.ARC2.new", Arrays.asList("Cryptodome.Cipher.ARC2.MODE_ECB", "Cryptodome.Cipher.ARC2.MODE_CBC"));
    CRYPTODOMEX_VULNERABLE_MODES_BY_API.put("Cryptodome.Cipher.Blowfish.new", Arrays.asList("Cryptodome.Cipher.Blowfish.MODE_ECB", "Cryptodome.Cipher.Blowfish.MODE_CBC"));
    CRYPTODOMEX_VULNERABLE_MODES_BY_API.put("Cryptodome.Cipher.AES.new", Arrays.asList("Cryptodome.Cipher.AES.MODE_ECB", "Cryptodome.Cipher.AES.MODE_CBC"));
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol == null) {
        return;
      }
      checkPycaLibrary(ctx, callExpression, calleeSymbol);
      checkPycryptoAndCryptodomeLibraries(ctx, callExpression, calleeSymbol);
      checkPydesLibrary(ctx, callExpression, calleeSymbol);
    });
  }

  private static void checkPydesLibrary(SubscriptionContext ctx, CallExpression callExpression, Symbol calleeSymbol) {
    if ("pyDes.des".equals(calleeSymbol.fullyQualifiedName())) {
      ctx.addIssue(callExpression.callee(), MESSAGE);
    }
  }

  private static void checkPycryptoAndCryptodomeLibraries(SubscriptionContext ctx, CallExpression callExpression, Symbol calleeSymbol) {
    if ("Crypto.Cipher.PKCS1_v1_5.new".equals(calleeSymbol.fullyQualifiedName()) || "Cryptodome.Cipher.PKCS1_v1_5.new".equals(calleeSymbol.fullyQualifiedName())) {
      ctx.addIssue(callExpression.callee(), MESSAGE);
      return;
    }

    List<String> vulnerableModes = PYCRYPTO_VULNERABLE_MODES_BY_API.getOrDefault(calleeSymbol.fullyQualifiedName(),
      CRYPTODOMEX_VULNERABLE_MODES_BY_API.get(calleeSymbol.fullyQualifiedName()));

    if (vulnerableModes != null) {
      RegularArgument mode = TreeUtils.nthArgumentOrKeyword(1, "mode", callExpression.arguments());
      if (mode != null) {
        Optional<Symbol> symbol = TreeUtils.getSymbolFromTree(mode.expression());
        symbol.filter(s -> vulnerableModes.contains(s.fullyQualifiedName())).ifPresent(s -> ctx.addIssue(mode, MESSAGE));
      }
    }
  }

  protected void checkPycaLibrary(SubscriptionContext ctx, CallExpression callExpression, Symbol calleeSymbol) {
    if ("cryptography.hazmat.primitives.ciphers.Cipher".equals(calleeSymbol.fullyQualifiedName())) {
      RegularArgument mode = TreeUtils.nthArgumentOrKeyword(1, "mode", callExpression.arguments());
      if (mode != null) {
        InferredType type = mode.expression().type();
        if (PYCA_VULNERABLE_MODES.stream().anyMatch(type::canOnlyBe)) {
          ctx.addIssue(mode, MESSAGE);
        }
      }
      return;
    }
    if (PYCA_RSA_KEY_METHODS.contains(calleeSymbol.fullyQualifiedName())) {
      RegularArgument padding = TreeUtils.nthArgumentOrKeyword(1, "padding", callExpression.arguments());
      if (padding != null && padding.expression().type().canOnlyBe("cryptography.hazmat.primitives.asymmetric.padding.PKCS1v15")) {
        ctx.addIssue(padding, MESSAGE);
      }
    }
  }
}
