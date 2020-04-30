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
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.REGULAR_ARGUMENT;

@Rule(key = "S5542")
public class SecureModeEncryptionAlgorithmsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use secure mode and padding scheme.";
  private static final Set<String> PYCA_ALGORITHM_METHODS = new HashSet<>(Arrays.asList(
    "cryptography.hazmat.primitives.ciphers.Cipher",
    "cryptography.hazmat.primitives.asymmetric.rsa.RSAPrivateKey.decrypt",
    "cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey.encrypt"
  ));

  private static final Set<String> PYCA_VULNERABLE_MODES_AND_PADDING_SCHEMES = new HashSet<>(Arrays.asList(
    "cryptography.hazmat.primitives.ciphers.modes.CBC",
    "cryptography.hazmat.primitives.ciphers.modes.ECB",
    "cryptography.hazmat.primitives.asymmetric.padding.PKCS1v15"
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
      List<Argument> arguments = callExpression.arguments();
      if (arguments.size() < 2) {
        return;
      }
      Argument argument = arguments.get(1);
      if (argument.is(REGULAR_ARGUMENT)) {
        Optional<Symbol> symbol = TreeUtils.getSymbolFromTree(((RegularArgument) argument).expression());
        symbol.filter(s -> vulnerableModes.contains(s.fullyQualifiedName())).ifPresent(s -> ctx.addIssue(argument, MESSAGE));
      }
    }
  }

  protected void checkPycaLibrary(SubscriptionContext ctx, CallExpression callExpression, Symbol calleeSymbol) {
    if (PYCA_ALGORITHM_METHODS.contains(calleeSymbol.fullyQualifiedName())) {
      List<Argument> arguments = callExpression.arguments();
      if (arguments.size() < 2) {
        return;
      }
      Argument argument = arguments.get(1);
      if (argument.is(REGULAR_ARGUMENT)) {
        InferredType type = ((RegularArgument) argument).expression().type();
        if (PYCA_VULNERABLE_MODES_AND_PADDING_SCHEMES.stream().anyMatch(type::canOnlyBe)) {
          ctx.addIssue(argument, MESSAGE);
        }
      }
    }
  }
}
