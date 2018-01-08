/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Collections;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonTokenType;

@Rule(key = LongIntegerWithLowercaseSuffixUsageCheck.CHECK_KEY)
public class LongIntegerWithLowercaseSuffixUsageCheck extends PythonCheck {

  public static final String CHECK_KEY = "LongIntegerWithLowercaseSuffixUsage";
  private static final String MESSAGE = "Replace suffix in long integers from lower case \"l\" to upper case \"L\".";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonTokenType.NUMBER);
  }

  @Override
  public void visitNode(AstNode astNode) {
    String value = astNode.getTokenValue();
    if (value.charAt(value.length() - 1) == 'l') {
      addIssue(astNode, MESSAGE);
    }
  }

}
