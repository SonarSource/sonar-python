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

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.RegularArgumentImpl;

// https://jira.sonarsource.com/browse/SONARPY-357
// https://jira.sonarsource.com/browse/RSPEC-4830
// https://jira.sonarsource.com/browse/MMF-1872
@Rule(key = "S4830")
public class VerifiedSslTlsCertificateCheck extends PythonSubscriptionCheck {

  private static final String VERIFY_NONE = Fqn.ssl("VERIFY_NONE");

  /**
   * Searches for `set_verify` invocations on instances of `OpenSSL.SSL.Context`,
   * extracts the flags from the first argument, checks that the combination of flags is secure.
   *
   * @param context {@inheritDoc}
   */
  @Override
  public void initialize(Context context) {

    final String setVerifyFqn = Fqn.context("set_verify");

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, subscriptionContext -> {
      CallExpression callExpr = (CallExpression) subscriptionContext.syntaxNode();

      boolean isSetVerifyInvocation = Optional
        .ofNullable(callExpr.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .filter(fqn -> fqn.equals(setVerifyFqn))
        .isPresent();

      if (isSetVerifyInvocation) {
        List<Argument> args = callExpr.arguments();
        if (!args.isEmpty()) {
          Tree flagsArgument = args.get(0);
          if (flagsArgument.is(Tree.Kind.REGULAR_ARGUMENT)) {
            Set<QualifiedExpression> flags = extractFlags(((RegularArgumentImpl) flagsArgument).expression());
            checkFlagSettings(flags).ifPresent(issue -> subscriptionContext.addIssue(issue.token, issue.message));
          }
        }
      }

    });
  }

  /** Helper methods for generating FQNs frequently used in this check. */
  private static class Fqn {
    private static String context(@SuppressWarnings("SameParameterValue") String method) {
      return ssl("Context." + method);
    }

    private static String ssl(String property) {
      return "OpenSSL.SSL." + property;
    }
  }

  /**
   * Recursively deconstructs binary trees of expressions separated with `|`-ors,
   * and collects the leafs that look like qualified expressions.
   */
  private static HashSet<QualifiedExpression> extractFlags(Tree flagsSubexpr) {
    if (flagsSubexpr.is(Tree.Kind.QUALIFIED_EXPR)) {
      // Base case: e.g. `SSL.VERIFY_NONE`
      return new HashSet<>(Collections.singletonList((QualifiedExpression) flagsSubexpr));
    } else if (flagsSubexpr.is(Tree.Kind.BITWISE_OR)) {
      // recurse into left and right branch
      BinaryExpression orExpr = (BinaryExpression) flagsSubexpr;
      HashSet<QualifiedExpression> flags = extractFlags(orExpr.leftOperand());
      flags.addAll(extractFlags(orExpr.rightOperand()));
      return flags;
    } else {
      // failed to interpret. Ignore leaf.
      return new HashSet<>();
    }
  }

  /**
   * Checks whether a combination of flags is valid,
   * optionally returns a message and a token if there is something wrong.
   */
  private static Optional<IssueReport> checkFlagSettings(Set<QualifiedExpression> flags) {
    for (QualifiedExpression qe : flags) {
      Symbol symb = qe.symbol();
      if (symb != null) {
        String fqn = symb.fullyQualifiedName();
        if (VERIFY_NONE.equals(fqn)) {
          return Optional.of(new IssueReport(
            "Omitting the check of the peer certificate is dangerous.",
            qe.lastToken()));
        }
      }
    }
    return Optional.empty();
  }

  /** Message and a token closest to the problematic position. Glorified <code>Pair&lt;A,B&gt;</code>. */
  private static class IssueReport {
    final String message;
    final Token token;

    private IssueReport(String message, Token token) {
      this.message = message;
      this.token = token;
    }
  }
}
