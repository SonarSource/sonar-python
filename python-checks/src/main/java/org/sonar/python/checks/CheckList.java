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

import com.google.common.collect.ImmutableList;

import java.util.List;

public final class CheckList {

  public static final String REPOSITORY_KEY = "python";

  public static final String SONAR_WAY_PROFILE = "Sonar way";

  private CheckList() {
  }

  public static List<Class> getChecks() {
    return ImmutableList.<Class>of(
      ParsingErrorCheck.class,
      CommentRegularExpressionCheck.class,
      LineLengthCheck.class,
      FunctionComplexityCheck.class,
      ClassComplexityCheck.class,
      FileComplexityCheck.class,
      NestedIfDepthCheck.class,
      OneStatementPerLineCheck.class,
      BackticksUsageCheck.class,
      InequalityUsageCheck.class,
      LongIntegerWithLowercaseSuffixUsageCheck.class,
      ExecStatementUsageCheck.class,
      PrintStatementUsageCheck.class,
      PreIncrementDecrementCheck.class,
      XPathCheck.class,
      TooManyLinesInFileCheck.class,
      ClassNameCheck.class,
      MissingDocstringCheck.class,
      FunctionNameCheck.class,
      MethodNameCheck.class,
      NewStyleClassCheck.class,
      UselessParenthesisAfterKeywordCheck.class,
      TooManyParametersCheck.class,
      NestedControlFlowDepthCheck.class,
      CollapsibleIfStatementsCheck.class,
      TrailingCommentCheck.class,
      BackslashInStringCheck.class,
      EmptyNestedBlockCheck.class);
  }

}
