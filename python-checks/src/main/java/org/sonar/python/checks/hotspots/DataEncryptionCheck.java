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

import java.util.Arrays;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.symbols.Symbol;

@Rule(key = "S4787")
public class DataEncryptionCheck extends PythonSubscriptionCheck {

  private static final List<String> FUNCTIONS_TO_CHECK = Arrays.asList(
    // pyca/cryptography: https://github.com/pyca/cryptography
    "cryptography.fernet.Fernet",
    "cryptography.hazmat.primitives.ciphers.aead.ChaCha20Poly1305",
    "cryptography.hazmat.primitives.ciphers.aead.AESGCM",
    "cryptography.hazmat.primitives.ciphers.aead.AESCCM",
    "cryptography.hazmat.primitives.asymmetric.dh.generate_parameters",
    "cryptography.hazmat.primitives.ciphers.Cipher",
    // pyca/pynacl: https://github.com/pyca/pynacl
    "nacl.public.Box",
    "nacl.secret.SecretBox",
    // PyCrypto: https://github.com/dlitz/pycrypto
    "Crypto.Cipher.AES.new",
    "Crypto.Cipher.DES.new",
    "Crypto.Cipher.DES3.new",
    "Crypto.Cipher.ARC2.new",
    "Crypto.Cipher.ARC4.new",
    "Crypto.Cipher.Blowfish.new",
    "Crypto.Cipher.CAST.new",
    "Crypto.Cipher.PKCS1_v1_5.new",
    "Crypto.Cipher.PKCS1_OAEP.new",
    "Crypto.Cipher.XOR.new",
    "Crypto.Cipher.XOR.new",
    "Crypto.PublicKey.ElGamal.generate",
    // Cryptodome: https://github.com/Legrandin/pycryptodome
    "Cryptodome.Cipher.AES.new",
    "Cryptodome.Cipher.ChaCha20.new",
    "Cryptodome.Cipher.DES.new",
    "Cryptodome.Cipher.DES3.new",
    "Cryptodome.Cipher.ARC2.new",
    "Cryptodome.Cipher.ARC4.new",
    "Cryptodome.Cipher.Blowfish.new",
    "Cryptodome.Cipher.CAST.new",
    "Cryptodome.Cipher.PKCS1_v1_5.new",
    "Cryptodome.Cipher.PKCS1_OAEP.new",
    "Cryptodome.Cipher.ChaCha20_Poly1305.new",
    "Cryptodome.Cipher.Salsa20.new",
    "Cryptodome.PublicKey.ElGamal.generate"
    );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol symbol = callExpression.calleeSymbol();
      if (symbol != null && FUNCTIONS_TO_CHECK.contains(symbol.fullyQualifiedName())) {
        ctx.addIssue(callExpression, "Make sure that encrypting data is safe here.");
      }
    });
  }
}
