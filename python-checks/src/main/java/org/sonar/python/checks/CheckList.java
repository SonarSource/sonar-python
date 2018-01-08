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

import org.sonar.python.PythonCheck;

public final class CheckList {

  public static final String REPOSITORY_KEY = "python";

  public static final String SONAR_WAY_PROFILE = "Sonar way";

  private CheckList() {
  }

  public static Iterable<Class> getChecks() {
    return PythonCheck.immutableSet(
        CognitiveComplexityFunctionCheck.class,
        ParsingErrorCheck.class,
        CommentRegularExpressionCheck.class,
        LineLengthCheck.class,
        FunctionComplexityCheck.class,
        ClassComplexityCheck.class,
        FileComplexityCheck.class,
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
        EmptyNestedBlockCheck.class,
        FixmeCommentCheck.class,
        SameConditionCheck.class,
        HardcodedIPCheck.class,
        NoPersonReferenceInTodoCheck.class,
        SameBranchCheck.class,
        BreakContinueOutsideLoopCheck.class,
        CommentedCodeCheck.class,
        ReturnYieldOutsideFunctionCheck.class,
        TrailingWhitespaceCheck.class,
        MissingNewlineAtEndOfFileCheck.class,
        LocalVariableAndParameterNameConventionCheck.class,
        InitReturnsValueCheck.class,
        ExitHasBadArgumentsCheck.class,
        ReturnAndYieldInOneFunctionCheck.class,
        ModuleNameCheck.class,
        FieldNameCheck.class,
        FieldDuplicatesClassNameCheck.class,
        UselessParenthesisCheck.class,
        DuplicatedMethodFieldNamesCheck.class,
        TooManyReturnsCheck.class,
        NeedlessPassCheck.class,
        UnusedLocalVariableCheck.class,
        AfterJumpStatementCheck.class,
        IdenticalExpressionOnBinaryOperatorCheck.class,
        SelfAssignmentCheck.class,
        MethodShouldBeStaticCheck.class
    );
  }

}
