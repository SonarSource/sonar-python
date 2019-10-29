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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import org.sonar.python.checks.hotspots.ClearTextProtocolsCheck;
import org.sonar.python.checks.hotspots.CommandLineArgsCheck;
import org.sonar.python.checks.hotspots.DataEncryptionCheck;
import org.sonar.python.checks.hotspots.DebugModeCheck;
import org.sonar.python.checks.hotspots.DisabledHtmlAutoEscapeCheck;
import org.sonar.python.checks.hotspots.DynamicCodeExecutionCheck;
import org.sonar.python.checks.hotspots.HashingDataCheck;
import org.sonar.python.checks.hotspots.LoggersConfigurationCheck;
import org.sonar.python.checks.hotspots.OsExecCheck;
import org.sonar.python.checks.hotspots.ProcessSignallingCheck;
import org.sonar.python.checks.hotspots.PseudoRandomCheck;
import org.sonar.python.checks.hotspots.PubliclyWritableDirectoriesCheck;
import org.sonar.python.checks.hotspots.RegexCheck;
import org.sonar.python.checks.hotspots.SQLQueriesCheck;
import org.sonar.python.checks.hotspots.StandardInputCheck;
import org.sonar.python.checks.hotspots.StrongCryptographicKeysCheck;

public final class CheckList {

  public static final String REPOSITORY_KEY = "python";

  private CheckList() {
  }

  public static Iterable<Class> getChecks() {
    return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
      AfterJumpStatementCheck.class,
      AllBranchesAreIdenticalCheck.class,
      BackslashInStringCheck.class,
      BackticksUsageCheck.class,
      BreakContinueOutsideLoopCheck.class,
      ClassComplexityCheck.class,
      ClassNameCheck.class,
      ClearTextProtocolsCheck.class,
      CognitiveComplexityFunctionCheck.class,
      CollapsibleIfStatementsCheck.class,
      CommandLineArgsCheck.class,
      CommentedCodeCheck.class,
      CommentRegularExpressionCheck.class,
      DataEncryptionCheck.class,
      DeadStoreCheck.class,
      DebugModeCheck.class,
      DisabledHtmlAutoEscapeCheck.class,
      DuplicatedMethodFieldNamesCheck.class,
      DuplicatedMethodImplementationCheck.class,
      DynamicCodeExecutionCheck.class,
      EmptyNestedBlockCheck.class,
      ExceptRethrowingCheck.class,
      ExecStatementUsageCheck.class,
      ExitHasBadArgumentsCheck.class,
      FieldDuplicatesClassNameCheck.class,
      FieldNameCheck.class,
      FileComplexityCheck.class,
      FixmeCommentCheck.class,
      FunctionComplexityCheck.class,
      FunctionNameCheck.class,
      HardcodedIPCheck.class,
      HashingDataCheck.class,
      IdenticalExpressionOnBinaryOperatorCheck.class,
      InequalityUsageCheck.class,
      InfiniteRecursionCheck.class,
      InitReturnsValueCheck.class,
      LineLengthCheck.class,
      LocalVariableAndParameterNameConventionCheck.class,
      LoggersConfigurationCheck.class,
      LongIntegerWithLowercaseSuffixUsageCheck.class,
      LoopExecutingAtMostOnceCheck.class,
      MethodNameCheck.class,
      MethodShouldBeStaticCheck.class,
      MissingDocstringCheck.class,
      MissingNewlineAtEndOfFileCheck.class,
      ModuleNameCheck.class,
      NeedlessPassCheck.class,
      NestedConditionalExpressionCheck.class,
      NestedControlFlowDepthCheck.class,
      NewStyleClassCheck.class,
      NoPersonReferenceInTodoCheck.class,
      OneStatementPerLineCheck.class,
      OsExecCheck.class,
      ParsingErrorCheck.class,
      PreIncrementDecrementCheck.class,
      PrintStatementUsageCheck.class,
      ProcessSignallingCheck.class,
      PseudoRandomCheck.class,
      PubliclyWritableDirectoriesCheck.class,
      RedundantJumpCheck.class,
      RegexCheck.class,
      ReturnAndYieldInOneFunctionCheck.class,
      ReturnYieldOutsideFunctionCheck.class,
      SameBranchCheck.class,
      SameConditionCheck.class,
      SelfAssignmentCheck.class,
      SQLQueriesCheck.class,
      StandardInputCheck.class,
      StringLiteralDuplicationCheck.class,
      StrongCryptographicKeysCheck.class,
      TempFileCreationCheck.class,
      TooManyLinesInFileCheck.class,
      TooManyParametersCheck.class,
      TooManyReturnsCheck.class,
      TrailingCommentCheck.class,
      TrailingWhitespaceCheck.class,
      UndeclaredNameUsageCheck.class,
      UnusedLocalVariableCheck.class,
      UselessParenthesisAfterKeywordCheck.class,
      UselessParenthesisCheck.class,
      WeakSSLProtocolCheck.class
    )));
  }

}
