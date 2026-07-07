/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.quickfix;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Stream;
import org.sonar.api.SonarProduct;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.IPythonTreeMaker;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public class PythonQuickFixVerifier {
  private static final String SEMANTIC_BASE_DIR = "/tmp/pythonQuickFixVerifier";

  private PythonQuickFixVerifier() {
  }

  public static void verify(PythonCheck check, String codeWithIssue, String... codesFixed) {
    verify(PythonQuickFixVerifier::createPythonVisitorContext, check, false, codeWithIssue, codesFixed);
  }

  public static void verifyNoQuickFixes(PythonCheck check, String codeWithIssue) {
    verifyNoQuickFixes(PythonQuickFixVerifier::createPythonVisitorContext, check, codeWithIssue);
  }

  public static void verifyQuickFixMessages(PythonCheck check, String codeWithIssue, String... expectedMessages) {
    verifyQuickFixMessages(PythonQuickFixVerifier::createPythonVisitorContext, check, codeWithIssue, expectedMessages);
  }

  public static void verifySemantic(PythonCheck check, String path, String codeWithIssue, String... codesFixed) {
    verify(code -> createSemanticVisitorContext(path, code), check, false, codeWithIssue, codesFixed);
  }

  public static void verifySemanticQuickFixMessages(PythonCheck check, String path, String codeWithIssue, String... expectedMessages) {
    verifyQuickFixMessages(code -> createSemanticVisitorContext(path, code), check, codeWithIssue, expectedMessages);
  }

  public static void verifySemanticNoQuickFixes(PythonCheck check, String path, String codeWithIssue) {
    verifyNoQuickFixes(code -> createSemanticVisitorContext(path, code), check, codeWithIssue);
  }

  public static void verifyIPython(PythonCheck check, String codeWithIssue, String... codesFixed) {
    verify(PythonQuickFixVerifier::createIPythonVisitorContext, check, true, codeWithIssue, codesFixed);
  }

  public static void verifyIPythonNoQuickFixes(PythonCheck check, String codeWithIssue) {
    verifyNoQuickFixes(PythonQuickFixVerifier::createIPythonVisitorContext, check, codeWithIssue);
  }

  public static void verifyIPythonQuickFixMessages(PythonCheck check, String codeWithIssue, String... expectedMessages) {
    verifyQuickFixMessages(PythonQuickFixVerifier::createIPythonVisitorContext, check, codeWithIssue, expectedMessages);
  }

  public static void verify(Function<String, PythonVisitorContext> createVisitorContext, PythonCheck check, boolean isIPython, String codeWithIssue, String... codesFixed) {
    List<PythonCheck.PreciseIssue> issues = PythonQuickFixVerifier
      .getIssuesWithQuickFix(createVisitorContext, check, codeWithIssue);

    assertThat(issues)
      .as("Number of issues")
      .overridingErrorMessage("Expected 1 issue but found %d", issues.size())
      .hasSize(1);
    PreciseIssue issue = issues.get(0);

    assertThat(issue.quickFixes())
      .as("Number of quickfixes")
      .overridingErrorMessage("Expected %d quickfix but found %d", codesFixed.length, issue.quickFixes().size())
      .hasSize(codesFixed.length);

    PythonParser pythonParser = isIPython ? PythonParser.createIPythonParser() : PythonParser.create();
    assertThatCode(() -> pythonParser.parse(String.join("\n", codesFixed))).as("Correction of quick fixes")
      .overridingErrorMessage("The code expected to be generated by the quickfix is not valid (I)Python code.\nResults is :\n%s", Arrays.asList(codesFixed))
      .doesNotThrowAnyException();

    List<String> appliedQuickFix = issue.quickFixes().stream()
      .map(quickFix -> applyQuickFix(codeWithIssue, quickFix))
      .toList();

    assertThat(appliedQuickFix)
      .as("""
        The code with the quickfix applied is not the expected result.
        Applied QuickFixes are:
        %s
        Expected result:
        %s""", appliedQuickFix, Arrays.asList(codesFixed))
      .isEqualTo(Arrays.asList(codesFixed));
  }

  public static void verifyNoQuickFixes(Function<String, PythonVisitorContext> createVisitorContext, PythonCheck check, String codeWithIssue) {
    List<PythonCheck.PreciseIssue> issues = PythonQuickFixVerifier
      .getIssuesWithQuickFix(createVisitorContext, check, codeWithIssue);

    assertThat(issues)
      .as("Number of issues")
      .overridingErrorMessage("Expected 1 issue but found %d", issues.size())
      .hasSize(1);
    PreciseIssue issue = issues.get(0);

    assertThat(issue.quickFixes())
      .as("Number of quick fixes")
      .overridingErrorMessage("Expected no quick fixes for the issue but found %d", issue.quickFixes().size())
      .isEmpty();
  }

  public static void verifyQuickFixMessages(Function<String, PythonVisitorContext> createVisitorContext,
    PythonCheck check,
    String codeWithIssue,
    String... expectedMessages) {
    Stream<String> descriptions = PythonQuickFixVerifier
      .getIssuesWithQuickFix(createVisitorContext, check, codeWithIssue)
      .stream()
      .flatMap(issue -> issue.quickFixes().stream())
      .map(PythonQuickFix::getDescription);

    assertThat(descriptions).containsExactly(expectedMessages);
  }

  private static List<PreciseIssue> scanFileForIssues(PythonCheck check, PythonVisitorContext context) {
    check.scanFile(context);
    if (check instanceof PythonSubscriptionCheck pythonSubscriptionCheck) {
      SubscriptionVisitor.analyze(Collections.singletonList(pythonSubscriptionCheck), context);
    }
    return context.getIssues();
  }

  private static List<PreciseIssue> getIssuesWithQuickFix(Function<String, PythonVisitorContext> createVisitorContext, PythonCheck check, String codeWithIssue) {
    var visitorContext = createVisitorContext.apply(codeWithIssue);
    return scanFileForIssues(check, visitorContext);
  }

  private static PythonVisitorContext createPythonVisitorContext(String code) {
    return createVisitorContext(PythonParser.create(), new PythonTreeMaker(), code);
  }

  private static PythonVisitorContext createIPythonVisitorContext(String code) {
    return createVisitorContext(PythonParser.createIPythonParser(), new IPythonTreeMaker(), code);
  }

  private static PythonVisitorContext createSemanticVisitorContext(String path, String code) {
    var pythonFile = new TestPythonVisitorRunner.MockPythonFile(SEMANTIC_BASE_DIR, path, code);
    ProjectLevelSymbolTable globalSymbols = TestPythonVisitorRunner.globalSymbols(Map.of(path, code), SEMANTIC_BASE_DIR);
    String packageName = pythonPackageName(pythonFile.file(), SEMANTIC_BASE_DIR);
    return TestPythonVisitorRunner.createContext(pythonFile, null, packageName, globalSymbols, CacheContextImpl.dummyCache());
  }

  private static PythonVisitorContext createVisitorContext(PythonParser parser, PythonTreeMaker treeMaker, String code) {
    var pythonFile = new PythonQuickFixFile(code);
    var astNode = parser.parse(pythonFile.content());
    var fileInput = treeMaker.fileInput(astNode);

    return new PythonVisitorContext.Builder(fileInput, pythonFile)
      .projectLevelSymbolTable(ProjectLevelSymbolTable.empty())
      .cacheContext(CacheContextImpl.dummyCache())
      .sonarProduct(SonarProduct.SONARLINT)
      .build();
  }

  private static String applyQuickFix(String codeWithIssue, PythonQuickFix quickFix) {
    List<PythonTextEdit> sortedEdits = sortTextEdits(quickFix.getTextEdits());
    String codeBeingFixed = codeWithIssue;
    for (PythonTextEdit edit : sortedEdits) {
      codeBeingFixed = applyTextEdit(codeBeingFixed, edit);
    }
    return codeBeingFixed;
  }

  private static String applyTextEdit(String codeWithIssue, PythonTextEdit textEdit) {
    String replacement = textEdit.replacementText();
    int start = convertPositionToIndex(codeWithIssue, textEdit.startLine(), textEdit.startLineOffset());
    int end = convertPositionToIndex(codeWithIssue, textEdit.endLine(), textEdit.endLineOffset());
    return codeWithIssue.substring(0, start) + replacement + codeWithIssue.substring(end);
  }

  private static List<PythonTextEdit> sortTextEdits(List<PythonTextEdit> pythonTextEdits) {
    checkNoCollision(pythonTextEdits);
    ArrayList<PythonTextEdit> list = new ArrayList<>(pythonTextEdits);
    list.sort(Comparator.comparingInt(PythonTextEdit::startLine).thenComparing(PythonTextEdit::startLineOffset));
    Collections.reverse(list);
    return Collections.unmodifiableList(list);
  }

  private static void checkNoCollision(List<PythonTextEdit> pythonTextEdits) throws IllegalArgumentException {
    for (int i = 0; i < pythonTextEdits.size(); i++) {
      PythonTextEdit edit = pythonTextEdits.get(i);
      for (int j = i + 1; j < pythonTextEdits.size(); j++) {
        PythonTextEdit edit2 = pythonTextEdits.get(j);
        if (oneEnclosedByTheOther(edit2, edit)) {
          throw new IllegalArgumentException("There is a collision between the range of the quickfixes.");
        }
      }
    }
  }

  // Returns true if the range of one edit crosses the range of the other. If one end of both edits is the same point,
  // we should return false
  private static boolean oneEnclosedByTheOther(PythonTextEdit toCheck, PythonTextEdit reference) {
    if (onSameLine(toCheck, reference)) {
      return sameLineIntersection(toCheck, reference);
    }
    if (compactOnDifferentLines(toCheck, reference)) {
      return false;
    }
    if (isCompact(toCheck)) {
      return isSecondInFirst(toCheck, reference);
    }
    if (isCompact(reference)) {
      return isSecondInFirst(reference, toCheck);
    }
    return explodedEditsIntersect(toCheck, reference);
  }

  private static boolean sameLineIntersection(PythonTextEdit toCheck, PythonTextEdit reference) {
    return !(toCheck.endLineOffset() < reference.startLineOffset() || toCheck.startLineOffset() > reference.endLineOffset());
  }

  private static boolean explodedEditsIntersect(PythonTextEdit toCheck, PythonTextEdit reference) {
    if (noLineIntersection(toCheck, reference)) {
      return false;
    }
    if (reference.startLine() == toCheck.endLine()) {
      return toCheck.endLineOffset() > reference.startLineOffset();
    }
    if (reference.endLine() == toCheck.startLine()) {
      return reference.endLineOffset() > toCheck.startLineOffset();
    }
    return true;
  }

  private static boolean onSameLine(PythonTextEdit check, PythonTextEdit ref) {
    return ref.startLine() == ref.endLine() && ref.endLine() == check.startLine() && check.startLine() == check.endLine();
  }

  private static boolean compactOnDifferentLines(PythonTextEdit check, PythonTextEdit ref) {
    return ref.startLine() == ref.endLine() && check.startLine() == check.endLine() && ref.endLine() != check.startLine();
  }

  private static boolean isCompact(PythonTextEdit check) {
    return check.startLine() == check.endLine();
  }

  // Returns true if there is an intersection between the ranges
  // The first parameter is a compact edit, i.e., the edit is only one one line
  private static boolean isSecondInFirst(PythonTextEdit first, PythonTextEdit second) {
    if (first.startLine() == second.startLine()) {
      return second.startLineOffset() < first.endLineOffset();
    } else if (first.endLine() == second.endLine()) {
      return first.startLineOffset() < second.endLineOffset();
    }
    // No intersection
    return false;
  }

  private static boolean noLineIntersection(PythonTextEdit check, PythonTextEdit ref) {
    return check.endLine() < ref.startLine() || check.startLine() > ref.endLine();
  }

  private static int convertPositionToIndex(String fileContent, int line, int lineOffset) {
    int currentLine = 1;
    int currentIndex = 0;
    while (currentLine < line) {
      currentIndex = fileContent.indexOf("\n", currentIndex) + 1;
      currentLine++;
    }
    return currentIndex + lineOffset;
  }

  private static class PythonQuickFixFile implements PythonFile {

    private final String content;

    public PythonQuickFixFile(String content) {
      this.content = content;
    }

    @Override
    public String content() {
      return this.content;
    }

    @Override
    public String fileName() {
      return "PythonQuickFixFile";
    }

    @Override
    public URI uri() {
      return URI.create(this.fileName());
    }

    @Override
    public String key() {
      return "PythonQuickFixFile";
    }
  }
}
