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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.TreeUtils;

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

import static java.util.Optional.ofNullable;

@Rule(key = XMLParserXXEVulnerableCheck.CHECK_KEY)
public class XMLParserXXEVulnerableCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "S2755";

  public static final String MESSAGE = "Remove or correct this useless self-assignment.";

  private static final String LXML_XMLPARSER = "lxml.etree.XMLParser";
  private static final String LXML_XSLT = "lxml.etree.XSLT";
  private static final String LXML_PARSE = "lxml.etree.parse";
  private static final String LXML_ACCESS_CONTROL = "lxml.etree.XSLTAccessControl";
  private static final String XML_SAX_MAKE_PARSER = "xml.sax.make_parser";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, XMLParserXXEVulnerableCheck::checkLxmlParseCall);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, XMLParserXXEVulnerableCheck::checkLxmlXsltCall);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, XMLParserXXEVulnerableCheck::checkSetFeatureCall);
  }

  private static boolean checkCallExpressionFqn(CallExpression callExpr, String fqn) {
    return ofNullable(callExpr.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(fqn::equals)
      .isPresent();
  }

  private static void checkLxmlParseCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (checkCallExpressionFqn(callExpression, LXML_PARSE)) {
      CallExpression parserCall = getParserCall(
        getArgValueAsCallExpression(
          TreeUtils.nthArgumentOrKeyword(1, "parser", callExpression.arguments())));
      if (parserCall != null && isUnsafeParserUsage(parserCall)) {
        ctx.addIssue(parserCall, MESSAGE).secondary(callExpression, MESSAGE);
      }
    }
  }

  @CheckForNull
  private static CallExpression getParserCall(@Nullable CallExpression callExpression) {
    if (callExpression != null && checkCallExpressionFqn(callExpression, LXML_XMLPARSER)) {
      return callExpression;
    }
    return null;
  }

  @CheckForNull
  private static CallExpression getArgValueAsCallExpression(@Nullable RegularArgument argument) {
    if (argument != null && argument.expression().is(Tree.Kind.NAME)) {
      Expression parserValue = Expressions.singleAssignedValue((NameImpl) argument.expression());
      if (parserValue != null && parserValue.is(Tree.Kind.CALL_EXPR)) {
        return (CallExpression) parserValue;
      }
    } else if (argument != null && argument.expression().is(Tree.Kind.CALL_EXPR)) {
      return (CallExpression) argument.expression();
    }
    return null;
  }

  private static boolean isUnsafeParserUsage(CallExpression callExpression) {
    RegularArgument noNetwork = TreeUtils.argumentByKeyword("no_network", callExpression.arguments());
    if (noNetwork != null && Expressions.isFalsy(noNetwork.expression())) {
      return true;
    }

    RegularArgument resolveEntities = TreeUtils.argumentByKeyword("resolve_entities", callExpression.arguments());
    return resolveEntities == null || !Expressions.isFalsy(resolveEntities.expression());
  }

  // XSLT

  private static void checkLxmlXsltCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (checkCallExpressionFqn(callExpression, LXML_XSLT)) {
      RegularArgument argument = TreeUtils.argumentByKeyword("access_control", callExpression.arguments());
      if (argument != null) {
        CallExpression xsltAclCall = getArgValueAsCallExpression(argument);
        if (xsltAclCall != null && checkCallExpressionFqn(xsltAclCall, LXML_ACCESS_CONTROL)) {
          RegularArgument readNetwork = TreeUtils.argumentByKeyword("read_network", xsltAclCall.arguments());
          if (readNetwork == null || !Expressions.isFalsy(readNetwork.expression())) {
            ctx.addIssue(xsltAclCall, MESSAGE).secondary(callExpression, MESSAGE);
          }
        }
      } else {
        ctx.addIssue(callExpression, MESSAGE);
      }
    }
  }

  // xml.sax

  private static void checkSetFeatureCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (isCallToSetFeature(callExpression) && checkSettingFeatureGesToTrue(callExpression)) {
      Expression makeParserCall = Expressions.singleAssignedValue((NameImpl) ((QualifiedExpression) callExpression.callee()).qualifier());
      if (makeParserCall != null &&
        makeParserCall.is(Tree.Kind.CALL_EXPR) &&
        checkCallExpressionFqn((CallExpression) makeParserCall, XML_SAX_MAKE_PARSER)) {
        ctx.addIssue(callExpression, MESSAGE).secondary(makeParserCall, MESSAGE);
      }
    }
  }

  private static boolean checkSettingFeatureGesToTrue(CallExpression callExpression) {
    if (callExpression.arguments().size() == 2) {
      RegularArgument first = (RegularArgument) callExpression.arguments().get(0);
      RegularArgument second = (RegularArgument) callExpression.arguments().get(1);
      return first.expression().is(Tree.Kind.NAME) &&
        "feature_external_ges".equals(((NameImpl) first.expression()).name()) &&
        !Expressions.isFalsy(second.expression());
    }
    return false;
  }

  private static boolean isCallToSetFeature(CallExpression callExpression) {
    return callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR) &&
      ((QualifiedExpression) callExpression.callee()).qualifier().is(Tree.Kind.NAME) &&
      ((QualifiedExpression) callExpression.callee()).name().name().equals("setFeature");
  }
}
