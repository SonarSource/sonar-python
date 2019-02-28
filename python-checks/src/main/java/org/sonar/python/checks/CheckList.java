/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import org.sonar.python.PythonCheck;
import org.sonar.python.checks.hotspots.CommandLineArgsCheck;
import org.sonar.python.checks.hotspots.OsExecCheck;
import org.sonar.python.checks.hotspots.ProcessSignallingCheck;

public final class CheckList {

  public static final String REPOSITORY_KEY = "python";

  private CheckList() {
  }

  public static Iterable<Class> getChecks() {
    return PythonCheck.immutableSet(
      AfterJumpStatementCheck.class,
      BackslashInStringCheck.class,
      BackticksUsageCheck.class,
      BreakContinueOutsideLoopCheck.class,
      ClassComplexityCheck.class,
      ClassNameCheck.class,
      CognitiveComplexityFunctionCheck.class,
      CollapsibleIfStatementsCheck.class,
      CommandLineArgsCheck.class,
      CommentedCodeCheck.class,
      CommentRegularExpressionCheck.class,
      DuplicatedMethodFieldNamesCheck.class,
      EmptyNestedBlockCheck.class,
      ExecStatementUsageCheck.class,
      ExitHasBadArgumentsCheck.class,
      FieldDuplicatesClassNameCheck.class,
      FieldNameCheck.class,
      FileComplexityCheck.class,
      FixmeCommentCheck.class,
      FunctionComplexityCheck.class,
      FunctionNameCheck.class,
      HardcodedIPCheck.class,
      IdenticalExpressionOnBinaryOperatorCheck.class,
      InequalityUsageCheck.class,
      InitReturnsValueCheck.class,
      LineLengthCheck.class,
      LocalVariableAndParameterNameConventionCheck.class,
      LongIntegerWithLowercaseSuffixUsageCheck.class,
      MethodNameCheck.class,
      MethodShouldBeStaticCheck.class,
      MissingDocstringCheck.class,
      MissingNewlineAtEndOfFileCheck.class,
      ModuleNameCheck.class,
      NeedlessPassCheck.class,
      NestedControlFlowDepthCheck.class,
      NewStyleClassCheck.class,
      NoPersonReferenceInTodoCheck.class,
      OneStatementPerLineCheck.class,
      OsExecCheck.class,
      ParsingErrorCheck.class,
      PreIncrementDecrementCheck.class,
      PrintStatementUsageCheck.class,
      ProcessSignallingCheck.class,
      ReturnAndYieldInOneFunctionCheck.class,
      ReturnYieldOutsideFunctionCheck.class,
      SameBranchCheck.class,
      SameConditionCheck.class,
      SelfAssignmentCheck.class,
      TooManyLinesInFileCheck.class,
      TooManyParametersCheck.class,
      TooManyReturnsCheck.class,
      TrailingCommentCheck.class,
      TrailingWhitespaceCheck.class,
      UnusedLocalVariableCheck.class,
      UselessParenthesisAfterKeywordCheck.class,
      UselessParenthesisCheck.class,
      XPathCheck.class
    );
  }

}
