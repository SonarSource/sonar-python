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

  private static String DES_MESSAGE = "DES works with 56-bit keys allow attacks via exhaustive search";
  private static String DES3_MESSAGE = "Triple DES is vulnerable to meet-in-the-middle attack";
  private static String RC2_MESSAGE = "RC2 is vulnerable to a related-key attack";
  private static String RC4_MESSAGE = "vulnerable to several attacks (see https://en.wikipedia.org/wiki/RC4#Security)";
  private static String BLOWFISH_MESSAGE = "Blowfish use a 64-bit block size makes it vulnerable to birthday attacks";

  // Idea is listed under "Weak Algorithms" in pyca/cryptography documentation
  // https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/\
  // #cryptography.hazmat.primitives.ciphers.algorithms.IDEA
  private static String IDEA_MESSAGE = "This cipher is susceptible to attacks when using weak keys. " +
    "Don't use for new applications.";

  static {

    // `pycryptodomex`, `pycryptodome`, and `pycrypto` all share the same names of the algorithms,
    // moreover, `pycryptodome` is drop-in replacement for `pycrypto`, thus they share same name ("Crypto").
    for (String libraryName : new String[] {"Cryptodome", "Crypto"}) {
      for (Map.Entry<String, String> e : (Iterable<Map.Entry<String, String>>) Stream.of(
        entry("DES", DES_MESSAGE),
        entry("DES3", DES3_MESSAGE),
        entry("ARC2", RC2_MESSAGE),
        entry("ARC4", RC4_MESSAGE),
        entry("Blowfish", BLOWFISH_MESSAGE))::iterator) {
        String methodName = e.getKey();
        String message = e.getValue();
        sensitiveCalleeFqnsAndMessages.put(String.format("%s.Cipher.%s.new", libraryName, methodName), message);
      }
    }

    // pyca (pyca/cryptography)
    for (Map.Entry<String, String> e : (Iterable<Map.Entry<String, String>>) Stream.of(
      entry("TripleDES", DES3_MESSAGE),
      entry("Blowfish", BLOWFISH_MESSAGE),
      entry("ARC4", RC4_MESSAGE),
      entry("IDEA", IDEA_MESSAGE))::iterator) {
      String methodName = e.getKey();
      String message = e.getValue();
      sensitiveCalleeFqnsAndMessages.put(
        String.format("cryptography.hazmat.primitives.ciphers.algorithms.%s", methodName),
        message);
    }

    // pydes
    sensitiveCalleeFqnsAndMessages.put("pyDes.des", DES_MESSAGE);
    sensitiveCalleeFqnsAndMessages.put("pyDes.triple_des", DES3_MESSAGE);
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
