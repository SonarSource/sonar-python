/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import java.util.stream.Stream;
import org.sonar.python.checks.cdk.DisabledEFSEncryptionCheck;
import org.sonar.python.checks.cdk.DisabledESDomainEncryptionCheck;
import org.sonar.python.checks.cdk.DisabledRDSEncryptionCheck;
import org.sonar.python.checks.cdk.DisabledSNSTopicEncryptionCheck;
import org.sonar.python.checks.cdk.IamPolicyPublicAccessCheck;
import org.sonar.python.checks.cdk.IamPrivilegeEscalationCheck;
import org.sonar.python.checks.cdk.NetworkCallsWithoutTimeoutsInLambdaCheck;
import org.sonar.python.checks.cdk.PrivilegePolicyCheck;
import org.sonar.python.checks.cdk.PublicApiIsSecuritySensitiveCheck;
import org.sonar.python.checks.cdk.PublicNetworkAccessToCloudResourcesCheck;
import org.sonar.python.checks.cdk.ResourceAccessPolicyCheck;
import org.sonar.python.checks.cdk.S3BucketBlockPublicAccessCheck;
import org.sonar.python.checks.cdk.S3BucketGrantedAccessCheck;
import org.sonar.python.checks.cdk.S3BucketHTTPCommunicationCheck;
import org.sonar.python.checks.cdk.S3BucketServerEncryptionCheck;
import org.sonar.python.checks.cdk.S3BucketVersioningCheck;
import org.sonar.python.checks.cdk.UnencryptedEbsVolumeCheck;
import org.sonar.python.checks.cdk.UnencryptedSageMakerNotebookCheck;
import org.sonar.python.checks.cdk.UnencryptedSqsQueueCheck;
import org.sonar.python.checks.cdk.UnrestrictedAdministrationCheck;
import org.sonar.python.checks.cdk.UnrestrictedOutboundCommunicationsCheck;
import org.sonar.python.checks.django.DjangoModelFormFieldsCheck;
import org.sonar.python.checks.django.DjangoModelStrMethodCheck;
import org.sonar.python.checks.django.DjangoModelStringFieldCheck;
import org.sonar.python.checks.django.DjangoReceiverDecoratorCheck;
import org.sonar.python.checks.hotspots.ClearTextProtocolsCheck;
import org.sonar.python.checks.hotspots.CommandLineArgsCheck;
import org.sonar.python.checks.hotspots.CorsCheck;
import org.sonar.python.checks.hotspots.CsrfDisabledCheck;
import org.sonar.python.checks.hotspots.DataEncryptionCheck;
import org.sonar.python.checks.hotspots.DebugModeCheck;
import org.sonar.python.checks.hotspots.DisabledHtmlAutoEscapeCheck;
import org.sonar.python.checks.hotspots.DisabledHtmlAutoEscapeLegacyCheck;
import org.sonar.python.checks.hotspots.DynamicCodeExecutionCheck;
import org.sonar.python.checks.hotspots.EmailSendingCheck;
import org.sonar.python.checks.hotspots.ExpandingArchiveCheck;
import org.sonar.python.checks.hotspots.FastHashingOrPlainTextCheck;
import org.sonar.python.checks.hotspots.GraphQLIntrospectionCheck;
import org.sonar.python.checks.hotspots.HardCodedCredentialsCheck;
import org.sonar.python.checks.hotspots.HardCodedCredentialsEntropyCheck;
import org.sonar.python.checks.hotspots.HashingDataCheck;
import org.sonar.python.checks.hotspots.HttpOnlyCookieCheck;
import org.sonar.python.checks.hotspots.LoggersConfigurationCheck;
import org.sonar.python.checks.hotspots.NonStandardCryptographicAlgorithmCheck;
import org.sonar.python.checks.hotspots.OsExecCheck;
import org.sonar.python.checks.hotspots.ProcessSignallingCheck;
import org.sonar.python.checks.hotspots.PseudoRandomCheck;
import org.sonar.python.checks.hotspots.PubliclyWritableDirectoriesCheck;
import org.sonar.python.checks.hotspots.RegexCheck;
import org.sonar.python.checks.hotspots.SQLQueriesCheck;
import org.sonar.python.checks.hotspots.SecureCookieCheck;
import org.sonar.python.checks.hotspots.StandardInputCheck;
import org.sonar.python.checks.hotspots.StrongCryptographicKeysCheck;
import org.sonar.python.checks.hotspots.UnsafeHttpMethodsCheck;
import org.sonar.python.checks.hotspots.UnverifiedHostnameCheck;
import org.sonar.python.checks.hotspots.XMLSignatureValidationCheck;
import org.sonar.python.checks.regex.AnchorPrecedenceCheck;
import org.sonar.python.checks.regex.DuplicatesInCharacterClassCheck;
import org.sonar.python.checks.regex.EmptyAlternativeCheck;
import org.sonar.python.checks.regex.EmptyGroupCheck;
import org.sonar.python.checks.regex.EmptyStringRepetitionCheck;
import org.sonar.python.checks.regex.GraphemeClustersInClassesCheck;
import org.sonar.python.checks.regex.GroupReplacementCheck;
import org.sonar.python.checks.regex.ImpossibleBackReferenceCheck;
import org.sonar.python.checks.regex.ImpossibleBoundariesCheck;
import org.sonar.python.checks.regex.InvalidRegexCheck;
import org.sonar.python.checks.regex.MultipleWhitespaceCheck;
import org.sonar.python.checks.regex.OctalEscapeCheck;
import org.sonar.python.checks.regex.PossessiveQuantifierContinuationCheck;
import org.sonar.python.checks.regex.RedosCheck;
import org.sonar.python.checks.regex.RedundantRegexAlternativesCheck;
import org.sonar.python.checks.regex.RegexComplexityCheck;
import org.sonar.python.checks.regex.RegexLookaheadCheck;
import org.sonar.python.checks.regex.ReluctantQuantifierCheck;
import org.sonar.python.checks.regex.ReluctantQuantifierWithEmptyContinuationCheck;
import org.sonar.python.checks.regex.SingleCharCharacterClassCheck;
import org.sonar.python.checks.regex.SingleCharacterAlternationCheck;
import org.sonar.python.checks.regex.StringReplaceCheck;
import org.sonar.python.checks.regex.SuperfluousCurlyBraceCheck;
import org.sonar.python.checks.regex.UnquantifiedNonCapturingGroupCheck;
import org.sonar.python.checks.regex.UnusedGroupNamesCheck;
import org.sonar.python.checks.regex.VerboseRegexCheck;
import org.sonar.python.checks.tests.AssertAfterRaiseCheck;
import org.sonar.python.checks.tests.AssertOnDissimilarTypesCheck;
import org.sonar.python.checks.tests.AssertOnTupleLiteralCheck;
import org.sonar.python.checks.tests.DedicatedAssertionCheck;
import org.sonar.python.checks.tests.ImplicitlySkippedTestCheck;
import org.sonar.python.checks.tests.NotDiscoverableTestMethodCheck;
import org.sonar.python.checks.tests.SkippedTestNoReasonCheck;
import org.sonar.python.checks.tests.UnconditionalAssertionCheck;

public class OpenSourceCheckList {

  public static final String RESOURCE_FOLDER = "org/sonar/l10n/py/rules/python";
  public static final String SONAR_WAY_PROFILE_LOCATION = RESOURCE_FOLDER + "/Sonar_way_profile.json";

  public Stream<Class<?>> getChecks() {
    return Stream.of(
      AfterJumpStatementCheck.class,
      AllBranchesAreIdenticalCheck.class,
      AnchorPrecedenceCheck.class,
      ArgumentNumberCheck.class,
      ArgumentTypeCheck.class,
      AssertOnDissimilarTypesCheck.class,
      AssertAfterRaiseCheck.class,
      AssertOnTupleLiteralCheck.class,
      AsyncFunctionNotAsyncCheck.class,
      AsyncFunctionWithTimeoutCheck.class,
      AsyncioTaskNotStoredCheck.class,
      AsyncAwsLambdaHandlerCheck.class,
      AsyncLongSleepCheck.class,
      AsyncWithContextManagerCheck.class,
      AwsClientExceptionNotCaughtCheck.class,
      AwsExpectedBucketOwnerCheck.class,
      AwsLambdaClientInstantiationCheck.class,
      AwsLambdaCrossCallCheck.class,
      AwsLambdaReservedEnvironmentVariableCheck.class,
      AwsCustomMetricNamespaceCheck.class,
      AwsHardcodedRegionCheck.class,
      AwsLambdaReturnValueAreSerializableCheck.class,
      AwsLambdaTmpCleanupCheck.class,
      AwsLongTermAccessKeysCheck.class,
      AwsMissingPaginationCheck.class,
      AwsWaitersInsteadOfCustomPollingCheck.class,
      BackslashInStringCheck.class,
      BackticksUsageCheck.class,
      BareRaiseInFinallyCheck.class,
      BooleanExpressionInExceptCheck.class,
      BooleanCheckNotInvertedCheck.class,
      BreakContinueOutsideLoopCheck.class,
      BuiltinShadowingAssignmentCheck.class,
      BuiltinGenericsOverTypingModuleCheck.class,
      BusyWaitingInAsyncCheck.class,
      CancellationScopeNoCheckpointCheck.class,
      CaughtExceptionsCheck.class,
      CancellationReraisedInAsyncCheck.class,
      ChangeMethodContractCheck.class,
      ChildAndParentExceptionCaughtCheck.class,
      CipherBlockChainingCheck.class,
      ClassComplexityCheck.class,
      ClassMethodFirstArgumentNameCheck.class,
      ClassNameCheck.class,
      ClearTextProtocolsCheck.class,
      CognitiveComplexityFunctionCheck.class,
      CollapsibleIfStatementsCheck.class,
      CollectionCreationWrappedInConstructorCheck.class,
      CollectionLengthComparisonCheck.class,
      CommandLineArgsCheck.class,
      CommentedCodeCheck.class,
      CommentRegularExpressionCheck.class,
      ComparisonToNoneCheck.class,
      CompressionModulesFromNamespaceCheck.class,
      ConfusingTypeCheckingCheck.class,
      ConfusingWalrusCheck.class,
      ConsistentReturnCheck.class,
      ConstantConditionCheck.class,
      ConstantValueDictComprehensionCheck.class,
      ControlFlowInTaskGroupCheck.class,
      CorsCheck.class,
      CsrfDisabledCheck.class,
      DataEncryptionCheck.class,
      DbNoPasswordCheck.class,
      DeadStoreCheck.class,
      DebugModeCheck.class,
      DedicatedAssertionCheck.class,
      DefaultFactoryArgumentCheck.class,
      DeprecatedNumpyTypesCheck.class,
      DictionaryDuplicateKeyCheck.class,
      DictionaryStaticKeyCheck.class,
      DirectTypeComparisonCheck.class,
      DisabledEFSEncryptionCheck.class,
      DisabledESDomainEncryptionCheck.class,
      DisabledHtmlAutoEscapeCheck.class,
      DisabledHtmlAutoEscapeLegacyCheck.class,
      DisabledRDSEncryptionCheck.class,
      DisabledSNSTopicEncryptionCheck.class,
      DjangoNonDictSerializationCheck.class,
      DjangoRenderContextCheck.class,
      DoublePrefixOperatorCheck.class,
      DuplicateArgumentCheck.class,
      DuplicatedMethodFieldNamesCheck.class,
      DuplicatedMethodImplementationCheck.class,
      DuplicatesInCharacterClassCheck.class,
      DynamicCodeExecutionCheck.class,
      EinopsSyntaxCheck.class,
      ElseAfterLoopsWithoutBreakCheck.class,
      EmailSendingCheck.class,
      EmptyAlternativeCheck.class,
      EmptyCollectionConstructorCheck.class,
      EmptyGroupCheck.class,
      EmptyFunctionCheck.class,
      EmptyNestedBlockCheck.class,
      EmptyStringRepetitionCheck.class,
      ExceptionGroupCheck.class,
      ExceptionCauseTypeCheck.class,
      ExceptionNotThrownCheck.class,
      ExceptionSuperClassDeclarationCheck.class,
      ExceptRethrowingCheck.class,
      ExecStatementUsageCheck.class,
      ExitHasBadArgumentsCheck.class,
      ExpandingArchiveCheck.class,
      FastHashingOrPlainTextCheck.class,
      FieldDuplicatesClassNameCheck.class,
      FieldNameCheck.class,
      FileComplexityCheck.class,
      FilePermissionsCheck.class,
      FileHeaderCopyrightCheck.class,
      FinallyBlockExitStatementCheck.class,
      FixmeCommentCheck.class,
      FlaskHardCodedJWTSecretKeyCheck.class,
      FlaskHardCodedSecretKeyCheck.class,
      FloatingPointEqualityCheck.class,
      FStringNestingLevelCheck.class,
      FunctionComplexityCheck.class,
      FunctionNameCheck.class,
      FunctionReturnTypeCheck.class,
      FunctionUsingLoopVariableCheck.class,
      GenericClassTypeParameterCheck.class,
      GenericExceptionRaisedCheck.class,
      GenericFunctionTypeParameterCheck.class,
      GenericTypeStatementCheck.class,
      GenericTypeWithoutArgumentCheck.class,
      GraphemeClustersInClassesCheck.class,
      GraphQLDenialOfServiceCheck.class,
      GraphQLIntrospectionCheck.class,
      GroupReplacementCheck.class,
      HardCodedCredentialsCheck.class,
      HardCodedCredentialsEntropyCheck.class,
      HardcodedIPCheck.class,
      HashMethodCheck.class,
      HashingDataCheck.class,
      HttpOnlyCookieCheck.class,
      IamPolicyPublicAccessCheck.class,
      IamPrivilegeEscalationCheck.class,
      IgnoredParameterCheck.class,
      IgnoredSystemExitCheck.class,
      IdenticalExpressionOnBinaryOperatorCheck.class,
      IdentityComparisonWithCachedTypesCheck.class,
      IdentityComparisonWithNewObjectCheck.class,
      IgnoredPureOperationsCheck.class,
      ImplicitStringConcatenationCheck.class,
      ImpossibleBackReferenceCheck.class,
      ImpossibleBoundariesCheck.class,
      IncompatibleOperandsCheck.class,
      InconsistentTypeHintCheck.class,
      IncorrectExceptionTypeCheck.class,
      IncorrectParameterDatetimeConstructorsCheck.class,
      IndexMethodCheck.class,
      InefficientDictIterationCheck.class,
      InequalityUsageCheck.class,
      InfiniteRecursionCheck.class,
      InitReturnsValueCheck.class,
      InstanceAndClassMethodsAtLeastOnePositionalCheck.class,
      InstanceMethodSelfAsFirstCheck.class,
      InputInAsyncCheck.class,
      InvalidOpenModeCheck.class,
      InvalidRegexCheck.class,
      InvariantReturnCheck.class,
      ItemOperationsTypeCheck.class,
      IterationOnNonIterableCheck.class,
      IterMethodReturnTypeCheck.class,
      IsCloseAbsTolCheck.class,
      JumpInFinallyCheck.class,
      JwtVerificationCheck.class,
      LambdaAssignmentCheck.class,
      LdapAuthenticationCheck.class,
      LineLengthCheck.class,
      LocalVariableAndParameterNameConventionCheck.class,
      LoggersConfigurationCheck.class,
      LongIntegerWithLowercaseSuffixUsageCheck.class,
      LoopExecutingAtMostOnceCheck.class,
      LoopOverDictKeyValuesCheck.class,
      MandatoryFunctionParameterTypeHintCheck.class,
      MandatoryFunctionReturnTypeHintCheck.class,
      MembershipTestSupportCheck.class,
      MethodNameCheck.class,
      MethodShouldBeStaticCheck.class,
      MissingDocstringCheck.class,
      MissingNewlineAtEndOfFileCheck.class,
      ModifiedParameterValueCheck.class,
      ModuleNameCheck.class,
      MultipleWhitespaceCheck.class,
      NeedlessPassCheck.class,
      NestedCollectionsCreationCheck.class,
      NestedConditionalExpressionCheck.class,
      NestedControlFlowDepthCheck.class,
      NewStyleClassCheck.class,
      NonCallableCalledCheck.class,
      NonStandardCryptographicAlgorithmCheck.class,
      NonStringInAllPropertyCheck.class,
      NonSingletonTfVariableCheck.class,
      NoPersonReferenceInTodoCheck.class,
      NoQaCommentCheck.class,
      NoReRaiseInExitCheck.class,
      NoSonarCommentCheck.class,
      NoSonarCommentFormatCheck.class,
      TemplateAndStrConcatenationCheck.class,
      NotDiscoverableTestMethodCheck.class,
      NotImplementedBooleanContextCheck.class,
      NotImplementedErrorInOperatorMethodsCheck.class,
      NumpyWeekMaskValidationCheck.class,
      NumpyIsNanCheck.class,
      NumpyListOverGeneratorCheck.class,
      NumpyRandomStateCheck.class,
      NumpyWhereOneConditionCheck.class,
      UnusedGroupNamesCheck.class,
      OctalEscapeCheck.class,
      OneStatementPerLineCheck.class,
      OsExecCheck.class,
      OverwrittenCollectionEntryCheck.class,
      PandasAddMergeParametersCheck.class,
      PandasChainInstructionCheck.class,
      PandasDataFrameToNumpyCheck.class,
      PandasModifyInPlaceCheck.class,
      PandasReadNoDataTypeCheck.class,
      PandasToDatetimeFormatCheck.class,
      ParsingErrorCheck.class,
      PredictableSaltCheck.class,
      PreIncrementDecrementCheck.class,
      PrintStatementUsageCheck.class,
      PrivilegePolicyCheck.class,
      ProcessSignallingCheck.class,
      PropertyAccessorParameterCountCheck.class,
      PytzUsageCheck.class,
      PseudoRandomCheck.class,
      PublicApiIsSecuritySensitiveCheck.class,
      PubliclyWritableDirectoriesCheck.class,
      PublicNetworkAccessToCloudResourcesCheck.class,
      PyTorchDataLoaderNumWorkersCheck.class,
      PytzTimeZoneInDatetimeConstructorCheck.class,
      RaiseOutsideExceptCheck.class,
      RandomSeedCheck.class,
      RedosCheck.class,
      RedundantJumpCheck.class,
      PossessiveQuantifierContinuationCheck.class,
      RedundantRegexAlternativesCheck.class,
      ReluctantQuantifierCheck.class,
      RegexCheck.class,
      ReluctantQuantifierWithEmptyContinuationCheck.class,
      ResourceAccessPolicyCheck.class,
      ReturnAndYieldInOneFunctionCheck.class,
      ReturnYieldOutsideFunctionCheck.class,
      RobustCipherAlgorithmCheck.class,
      S3BucketBlockPublicAccessCheck.class,
      S3BucketGrantedAccessCheck.class,
      S3BucketHTTPCommunicationCheck.class,
      S3BucketServerEncryptionCheck.class,
      S3BucketVersioningCheck.class,
      SameBranchCheck.class,
      SameConditionCheck.class,
      SecureCookieCheck.class,
      SecureModeEncryptionAlgorithmsCheck.class,
      SelfAssignmentCheck.class,
      SetDuplicateKeyCheck.class,
      SideEffectInTfFunctionCheck.class,
      SillyEqualityCheck.class,
      SillyIdentityCheck.class,
      SingleCharacterAlternationCheck.class,
      SingleCharCharacterClassCheck.class,
      SkippedTestNoReasonCheck.class,
      SkLearnEstimatorDontInitializeEstimatedValuesCheck.class,
      SleepZeroInAsyncCheck.class,
      SpecialMethodParamListCheck.class,
      SpecialMethodReturnTypeCheck.class,
      SQLQueriesCheck.class,
      StandardInputCheck.class,
      StrftimeConfusingHourSystemCheck.class,
      StringFormatCorrectnessCheck.class,
      StringFormatMisuseCheck.class,
      StringLiteralDuplicationCheck.class,
      StringReplaceCheck.class,
      StrongCryptographicKeysCheck.class,
      SklearnCachedPipelineDontAccessTransformersCheck.class,
      MissingHyperParameterCheck.class,
      NetworkCallsWithoutTimeoutsInLambdaCheck.class,
      SklearnPipelineSpecifyMemoryArgumentCheck.class,
      SklearnPipelineParameterAreCorrectCheck.class,
      SuperfluousCurlyBraceCheck.class,
      SynchronousFileOperationsInAsyncCheck.class,
      SynchronousHttpOperationsInAsyncCheck.class,
      SynchronousSubprocessOperationsInAsyncCheck.class,
      SynchronousOsCallsInAsyncCheck.class,
      TempFileCreationCheck.class,
      ImplicitlySkippedTestCheck.class,
      TaskGroupNurseryUsedOnlyOnceCheck.class,
      TemplateStringStructuralPatternMatchingCheck.class,
      TimeSleepInAsyncCheck.class,
      ToDoCommentCheck.class,
      TooManyLinesInFileCheck.class,
      TooManyLinesInFunctionCheck.class,
      TooManyParametersCheck.class,
      TooManyReturnsCheck.class,
      TorchAutogradVariableShouldNotBeUsedCheck.class,
      TorchLoadLeadsToUntrustedCodeExecutionCheck.class,
      TorchModuleModeShouldBeSetAfterLoadingCheck.class,
      TorchModuleShouldCallInitCheck.class,
      TrailingCommentCheck.class,
      TrailingWhitespaceCheck.class,
      TypeAliasAnnotationCheck.class,
      TfFunctionDependOnOutsideVariableCheck.class,
      TfFunctionRecursivityCheck.class,
      TfInputShapeOnModelSubclassCheck.class,
      TfGatherDeprecatedValidateIndicesCheck.class,
      TfPyTorchSpecifyReductionAxisCheck.class,
      ReferencedBeforeAssignmentCheck.class,
      RegexComplexityCheck.class,
      RegexLookaheadCheck.class,
      RewriteCollectionConstructorAsComprehensionCheck.class,
      UnconditionalAssertionCheck.class,
      UndefinedNameAllPropertyCheck.class,
      UnencryptedSageMakerNotebookCheck.class,
      UnencryptedSqsQueueCheck.class,
      UnencryptedEbsVolumeCheck.class,
      UnionTypeExpressionCheck.class,
      UnnecessaryListComprehensionArgumentCheck.class,
      UnnecessaryComprehensionCheck.class,
      UnnecessaryLambdaMapCallCheck.class,
      UnnecessaryListCastCheck.class,
      UnnecessaryReversedCallCheck.class,
      UnnecessarySortInsideSetCheck.class,
      UnnecessarySubscriptReversalCheck.class,
      UnprocessedTemplateStringCheck.class,
      UnquantifiedNonCapturingGroupCheck.class,
      UnreachableExceptCheck.class,
      UnreadPrivateAttributesCheck.class,
      UnreadPrivateInnerClassesCheck.class,
      UnreadPrivateMethodsCheck.class,
      UnrestrictedAdministrationCheck.class,
      UnrestrictedOutboundCommunicationsCheck.class,
      UnsafeHttpMethodsCheck.class,
      UndefinedSymbolsCheck.class,
      UnusedFunctionParameterCheck.class,
      UnusedImportCheck.class,
      UnusedLocalVariableCheck.class,
      UnusedNestedDefinitionCheck.class,
      UnverifiedHostnameCheck.class,
      UselessParenthesisAfterKeywordCheck.class,
      UselessParenthesisCheck.class,
      UselessStatementCheck.class,
      UseOfAnyAsTypeHintCheck.class,
      UseOfEmptyReturnValueCheck.class,
      TimezoneNaiveDatetimeConstructorsCheck.class,
      UseStartsWithEndsWithCheck.class,
      VerboseRegexCheck.class,
      VerifiedSslTlsCertificateCheck.class,
      WeakSSLProtocolCheck.class,
      WildcardImportCheck.class,
      WrongAssignmentOperatorCheck.class,
      XMLParserXXEVulnerableCheck.class,
      XMLSignatureValidationCheck.class,
      DjangoModelFormFieldsCheck.class,
      DjangoReceiverDecoratorCheck.class,
      DjangoModelStringFieldCheck.class,
      DjangoModelStrMethodCheck.class,
      HardcodedCredentialsCallCheck.class);
  }
}
