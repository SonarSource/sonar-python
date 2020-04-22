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

import java.util.AbstractMap;
import java.util.HashMap;
import java.util.Map;

import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

// https://jira.sonarsource.com/browse/RSPEC-5547 (general)
// https://jira.sonarsource.com/browse/RSPEC-5552 (python-specific)
@Rule(key = "S5547")
public class RobustCipherAlgorithmCheck extends PythonSubscriptionCheck {

  private static final Map<String, String> sensitiveCalleeFqnsAndMessages = new HashMap<>();

  static {
    String desMessage = "DES works with 56-bit keys that allow attacks via exhaustive search";
    String des3Message = "Triple DES is vulnerable to meet-in-the-middle attacks";
    String rc2Message = "RC2 is vulnerable to a related-key attack";
    String rc4Message = "RC4 is vulnerable to several attacks";
    String blowfishMessage = "Blowfish uses a 64-bit block size, which makes it vulnerable to birthday attacks";

    // Idea is listed under "Weak Algorithms" in pyca/cryptography documentation
    // https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/\
    // #cryptography.hazmat.primitives.ciphers.algorithms.IDEA
    String ideaMessage = "IDEA-cipher is susceptible to attacks when using weak keys";

    // `pycryptodomex`, `pycryptodome`, and `pycrypto` all share the same names of the algorithms,
    // moreover, `pycryptodome` is drop-in replacement for `pycrypto`, thus they share same name ("Crypto").
    for (String libraryName : new String[] {"Cryptodome", "Crypto"}) {
      for (Map.Entry<String, String> e : (Iterable<Map.Entry<String, String>>) Stream.of(
        entry("DES", desMessage),
        entry("DES3", des3Message),
        entry("ARC2", rc2Message),
        entry("ARC4", rc4Message),
        entry("Blowfish", blowfishMessage))::iterator) {
        String methodName = e.getKey();
        String message = e.getValue();
        sensitiveCalleeFqnsAndMessages.put(String.format("%s.Cipher.%s.new", libraryName, methodName), message);
      }
    }

    // pyca (pyca/cryptography)
    for (Map.Entry<String, String> e : (Iterable<Map.Entry<String, String>>) Stream.of(
      entry("TripleDES", des3Message),
      entry("Blowfish", blowfishMessage),
      entry("ARC4", rc4Message),
      entry("IDEA", ideaMessage))::iterator) {
      String methodName = e.getKey();
      String message = e.getValue();
      sensitiveCalleeFqnsAndMessages.put(
        String.format("cryptography.hazmat.primitives.ciphers.algorithms.%s", methodName),
        message);
    }

    // pydes
    sensitiveCalleeFqnsAndMessages.put("pyDes.des", desMessage);
    sensitiveCalleeFqnsAndMessages.put("pyDes.triple_des", des3Message);
  }

  /** Pair constructor. */
  private static <K, V> Map.Entry<K, V> entry(K key, V value) {
    // It's not uncommon: https://www.baeldung.com/java-initialize-hashmap#the-java-8-way
    return new AbstractMap.SimpleEntry<>(key, value);
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, subscriptionContext -> {
      CallExpression callExpr = (CallExpression) subscriptionContext.syntaxNode();
      Symbol calleeSymbol = callExpr.calleeSymbol();
      if (calleeSymbol != null) {
        String fqn = calleeSymbol.fullyQualifiedName();
        if (fqn != null && sensitiveCalleeFqnsAndMessages.containsKey(fqn)) {
          subscriptionContext.addIssue(callExpr.callee(), sensitiveCalleeFqnsAndMessages.get(fqn));
        }
      }
    });
  }

}
