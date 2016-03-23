/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonTokenType;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Rule(
    key = HardcodedIPCheck.CHECK_KEY,
    priority = Priority.MAJOR,
    name = "IP addresses should not be hardcoded",
    tags = {Tags.CERT, Tags.SECURITY}
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.ARCHITECTURE_CHANGEABILITY)
@SqaleConstantRemediation("30min")
@ActivatedByDefault
public class HardcodedIPCheck extends SquidCheck<Grammar> {
  public static final String CHECK_KEY = "S1313";

  private static final String IP_ADDRESS_V4_REGEX = "((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?))";
  private static final String IP_ADDRESS_V6_REGEX = "((::)?([\\da-fA-F]{1,4}::?){2,7})([\\da-fA-F]{1,4})?";
  private static final Pattern patternV4 = Pattern.compile(IP_ADDRESS_V4_REGEX);
  private static final Pattern patternV6 = Pattern.compile(IP_ADDRESS_V6_REGEX);
  String message = "Make this IP \"%s\" address configurable.";

  @Override
  public void init() {
    subscribeTo(PythonTokenType.STRING);
  }

  @Override
  public void visitNode(AstNode node) {
    String string = node.getTokenOriginalValue();
    if (isMultilineString(string)) {
      return;
    }

    Matcher matcher = patternV4.matcher(string);

    if (matcher.find()) {
      String ipAddress = matcher.group();
      getContext().createLineViolation(this, String.format(message, ipAddress), node);
    } else {
      matcher = patternV6.matcher(string);
      if (matcher.find()) {
        String ipAddress = matcher.group();
        if (ipAddress.length() > 8) {
          getContext().createLineViolation(this, String.format(message, ipAddress), node);
        }
      }
    }
  }

  private boolean isMultilineString(String string) {
    return string.endsWith("'''") || string.endsWith("\"\"\"");
  }
}

