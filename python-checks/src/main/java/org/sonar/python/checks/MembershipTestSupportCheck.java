/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.List;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.InExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5642")
public class MembershipTestSupportCheck extends PythonSubscriptionCheck {
  private static final String PRIMARY_MESSAGE = "Change the type of %s";
  private static final String PRIMARY_MESSAGE_MULTILINE = "Change the type for the target expression of `in`";
  private static final String KNOWN_TYPE_MESSAGE = "; type %s does not support membership protocol.";
  private static final String UNKNOWN_TYPE_MESSAGE = "; the type does not support the membership protocol.";
  private static final String SECONDARY_MESSAGE = "The result value of this expression does not support the membership protocol.";

  // The ordering of membership protocol methods matters here.
  // For instance, a class that has __contains__ set to None but which defines __iter__ does not fulfill the membership protocol
  // A class that defines __contains__ and has __iter__ set to None does fulfill the membership protocol
  private static final List<String> MEMBERSHIP_PROTOCOL_ENABLING_METHODS = List.of("__contains__", "__iter__", "__getitem__");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IN, ctx -> checkInExpression(ctx, (InExpression) ctx.syntaxNode()));
  }

  private static void checkInExpression(SubscriptionContext ctx, InExpression inExpression) {
    Expression rhs = inExpression.rightOperand();
    InferredType rhsType = rhs.type();

    if (canSupportMembershipProtocol(rhsType)) {
      return;
    }

    PreciseIssue primaryLocation = addPrimaryIssue(ctx, inExpression, genPrimaryMessage(rhs, rhsType));
    primaryLocation.secondary(inExpression.rightOperand(), SECONDARY_MESSAGE);
  }

  private static String genPrimaryMessage(Expression rhs, InferredType rhsType) {
    final String typeMessage;
    Symbol typeSymbol = rhsType.runtimeTypeSymbol();
    if (typeSymbol != null) {
      typeMessage = String.format(KNOWN_TYPE_MESSAGE, typeSymbol.fullyQualifiedName());
    } else {
      typeMessage = UNKNOWN_TYPE_MESSAGE;
    }

    String inTarget = TreeUtils.treeToString(rhs, false);
    String message;
    if (inTarget == null) {
      message = PRIMARY_MESSAGE_MULTILINE;
    } else {
      message = String.format(PRIMARY_MESSAGE, inTarget);
    }
    message += typeMessage;

    return message;
  }

  private static PreciseIssue addPrimaryIssue(SubscriptionContext ctx, InExpression inExpression, String message) {
    var notToken = inExpression.notToken();
    if (notToken == null) {
      return ctx.addIssue(inExpression.operator(), message);
    }

    return ctx.addIssue(notToken, inExpression.operator(), message);
  }

  private static boolean canSupportMembershipProtocol(InferredType type) {
    boolean atLeastOneMemberUnknown = false;
    for (var methodName : MEMBERSHIP_PROTOCOL_ENABLING_METHODS) {
      switch (memberType(type, methodName)) {
        case NOT_A_METHOD:
          return false;
        case METHOD:
          return true;
        case UNKNOWN:
          atLeastOneMemberUnknown = true;
          break;
        default:
      }
    }

    return atLeastOneMemberUnknown;
  }

  private enum MemberType {
    METHOD,
    NOT_A_METHOD,
    NOT_PRESENT,
    UNKNOWN
  }

  private static boolean canBeMethodSymbol(Symbol symbol) {
    if (symbol.is(Symbol.Kind.FUNCTION)) {
      return true;
    }

    // To avoid FPs, we accept OTHER unless we can show that it is a non-callable.
    //
    // This handles cases like __contains__ = other() or __contains__ = None.
    // Although it is unclear, whether such edge cases really appear in the wild
    if (symbol.is(Symbol.Kind.OTHER)) {
      // if we can definitely show that the symbol refers to a non-callable, we return false
      var bindingUsages = symbol.usages().stream().filter(Usage::isBindingUsage).limit(2).collect(Collectors.toUnmodifiableList());
      if (bindingUsages.size() == 1) {
        var bindingUsage = bindingUsages.get(0);
        var assignment = TreeUtils.firstAncestorOfKind(bindingUsage.tree(), Tree.Kind.ASSIGNMENT_STMT);

        return assignment == null || ((AssignmentStatement) assignment).assignedValue().type().canHaveMember("__call__");
      }

      // otherwise, we always accept OTHER symbols
      return true;
    }

    if (symbol.is(Symbol.Kind.AMBIGUOUS)) {
      return ((AmbiguousSymbol) symbol).alternatives().stream().anyMatch(MembershipTestSupportCheck::canBeMethodSymbol);
    }

    return false;
  }

  private static MemberType memberType(InferredType type, String methodName) {
    var maybeMember = type.resolveMember(methodName);
    if (maybeMember.isPresent()) {
      var symbol = maybeMember.get();

      if (canBeMethodSymbol(symbol)) {
        return MemberType.METHOD;
      }

      return MemberType.NOT_A_METHOD;
    }

    if (!type.canHaveMember(methodName)) {
      return MemberType.NOT_PRESENT;
    }

    return MemberType.UNKNOWN;
  }
}
