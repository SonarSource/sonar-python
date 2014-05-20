/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.Grammar;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.checks.SquidCheck;
import org.sonar.sslr.grammar.GrammarRuleKey;

/**
 * Note that implementation differs from AbstractNestedIfCheck - see SONARPLUGINS-1855
 */
@Rule(
  key = "NestedIfDepth",
  priority = Priority.MINOR,
  status = org.sonar.api.rules.Rule.STATUS_DEPRECATED)
public class NestedIfDepthCheck extends SquidCheck<Grammar> {

  private int nestingLevel;

  private static final int DEFAULT_MAXIMUM_NESTING_LEVEL = 3;

  @RuleProperty(
    key = "maximumNestingLevel",
    defaultValue = "" + DEFAULT_MAXIMUM_NESTING_LEVEL)
  public int maximumNestingLevel = DEFAULT_MAXIMUM_NESTING_LEVEL;

  public int getMaximumNestingLevel() {
    return maximumNestingLevel;
  }

  public GrammarRuleKey getIfRule() {
    return PythonGrammar.IF_STMT;
  }

  @Override
  public void init() {
    subscribeTo(getIfRule());
  }

  @Override
  public void visitNode(AstNode astNode) {
    nestingLevel++;
    if (nestingLevel == getMaximumNestingLevel() + 1) {
      getContext().createLineViolation(this, "This if has a nesting level of {0}, which is higher than the maximum allowed {1}.", astNode,
        nestingLevel,
        getMaximumNestingLevel());
    }
  }

  @Override
  public void leaveNode(AstNode astNode) {
    nestingLevel--;
  }

}
