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

import java.util.HashMap;
import java.util.Map;
import static java.util.Arrays.asList;

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
    String issueMessage = "Use a strong cipher algorithm.";


    // `pycryptodomex`, `pycryptodome`, and `pycrypto` all share the same names of the algorithms,
    // moreover, `pycryptodome` is drop-in replacement for `pycrypto`, thus they share same name ("Crypto").
    for (String libraryName : asList("Cryptodome", "Crypto")) {
      for (String vulnerableMethodName : asList("DES", "DES3", "ARC2", "ARC4", "Blowfish")) {
        sensitiveCalleeFqnsAndMessages.put(String.format("%s.Cipher.%s.new", libraryName, vulnerableMethodName), issueMessage);
      }
    }


    // Idea is listed under "Weak Algorithms" in pyca/cryptography documentation
    // https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/\
    // #cryptography.hazmat.primitives.ciphers.algorithms.IDEA
    // pyca (pyca/cryptography)
    for (String methodName : asList("TripleDES", "Blowfish", "ARC4", "IDEA")) {
      sensitiveCalleeFqnsAndMessages.put(
        String.format("cryptography.hazmat.primitives.ciphers.algorithms.%s", methodName), issueMessage);
    }

    // pydes
    sensitiveCalleeFqnsAndMessages.put("pyDes.des", issueMessage);
    sensitiveCalleeFqnsAndMessages.put("pyDes.triple_des", issueMessage);
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
