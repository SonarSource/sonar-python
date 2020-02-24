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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import org.sonar.python.checks.hotspots.ClearTextProtocolsCheck;
import org.sonar.python.checks.hotspots.CommandLineArgsCheck;
import org.sonar.python.checks.hotspots.CorsCheck;
import org.sonar.python.checks.hotspots.DataEncryptionCheck;
import org.sonar.python.checks.hotspots.DebugModeCheck;
import org.sonar.python.checks.hotspots.DisabledHtmlAutoEscapeCheck;
import org.sonar.python.checks.hotspots.DynamicCodeExecutionCheck;
import org.sonar.python.checks.hotspots.EmailSendingCheck;
import org.sonar.python.checks.hotspots.ExpandingArchiveCheck;
import org.sonar.python.checks.hotspots.HardCodedCredentialsCheck;
import org.sonar.python.checks.hotspots.HashingDataCheck;
import org.sonar.python.checks.hotspots.HttpOnlyCookieCheck;
import org.sonar.python.checks.hotspots.LoggersConfigurationCheck;
import org.sonar.python.checks.hotspots.OsExecCheck;
import org.sonar.python.checks.hotspots.ProcessSignallingCheck;
import org.sonar.python.checks.hotspots.PseudoRandomCheck;
import org.sonar.python.checks.hotspots.PubliclyWritableDirectoriesCheck;
import org.sonar.python.checks.hotspots.RegexCheck;
import org.sonar.python.checks.hotspots.SQLQueriesCheck;
import org.sonar.python.checks.hotspots.SecureCookieCheck;
import org.sonar.python.checks.hotspots.StandardInputCheck;
import org.sonar.python.checks.hotspots.StrongCryptographicKeysCheck;
import org.sonar.python.checks.hotspots.UnverifiedHostnameCheck;

public final class CheckList {

  public static final String REPOSITORY_KEY = "python";

  private CheckList() {
  }

  public static Iterable<Class> getChecks() {
    return Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
      AfterJumpStatementCheck.class,
      AllBranchesAreIdenticalCheck.class,
      ArgumentNumberCheck.class,
      BackslashInStringCheck.class,
      BackticksUsageCheck.class,
      BreakContinueOutsideLoopCheck.class,
      ClassComplexityCheck.class,
      ClassNameCheck.class,
      ClearTextProtocolsCheck.class,
      CognitiveComplexityFunctionCheck.class,
      CollapsibleIfStatementsCheck.class,
      CollectionLengthComparisonCheck.class,
      CommandLineArgsCheck.class,
      CommentedCodeCheck.class,
      CommentRegularExpressionCheck.class,
      CorsCheck.class,
      DataEncryptionCheck.class,
      DbNoPasswordCheck.class,
      DeadStoreCheck.class,
      DebugModeCheck.class,
      DisabledHtmlAutoEscapeCheck.class,
      DuplicatedMethodFieldNamesCheck.class,
      DuplicatedMethodImplementationCheck.class,
      DynamicCodeExecutionCheck.class,
      EmailSendingCheck.class,
      EmptyFunctionCheck.class,
      EmptyNestedBlockCheck.class,
      ExceptRethrowingCheck.class,
      ExecStatementUsageCheck.class,
      ExitHasBadArgumentsCheck.class,
      ExpandingArchiveCheck.class,
      FieldDuplicatesClassNameCheck.class,
      FieldNameCheck.class,
      FileComplexityCheck.class,
      FixmeCommentCheck.class,
      FunctionComplexityCheck.class,
      FunctionNameCheck.class,
      HardCodedCredentialsCheck.class,
      HardcodedIPCheck.class,
      HashingDataCheck.class,
      HttpOnlyCookieCheck.class,
      IgnoredParameterCheck.class,
      IdenticalExpressionOnBinaryOperatorCheck.class,
      IncorrectExceptionTypeCheck.class,
      InequalityUsageCheck.class,
      InfiniteRecursionCheck.class,
      InitReturnsValueCheck.class,
      InvariantReturnCheck.class,
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
      OverwrittenCollectionEntryCheck.class,
      ParsingErrorCheck.class,
      PredictableSaltCheck.class,
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
      SecureCookieCheck.class,
      SelfAssignmentCheck.class,
      SillyEqualityCheck.class,
      SillyIdentityCheck.class,
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
      UnusedNestedDefinitionCheck.class,
      UnverifiedHostnameCheck.class,
      UselessParenthesisAfterKeywordCheck.class,
      UselessParenthesisCheck.class,
      WeakSSLProtocolCheck.class,
      WrongAssignmentOperatorCheck.class
    )));
  }

}
