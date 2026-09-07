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
package org.sonar.plugins.python;

import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.checks.TrailingWhitespaceCheck;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.parser.IPythonParserConfiguration.OpaqueCellRange;
import org.sonar.python.tree.IPythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.plugins.python.TestUtils.createInputFile;

class IpynbNotebookParserScannerTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python");
  private final File frontendBaseDir = new File("../python-frontend/src/test/resources/org/sonar/plugins/python");

  @Test
  void generated_notebook_python_uses_the_test_file_classifier() {
    var inputFile = createInputFile(baseDir, "notebook_trailing_whitespace.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var notebook = IpynbNotebookParser.parseNotebook(inputFile).get();
    var generatedFile = new GeneratedIPythonFile(notebook.wrappedFile(), "import pytest\n", Map.of());
    var context = new PythonVisitorContext.Builder(
      TestPythonVisitorRunner.parseNotebookFile(Map.of(), "import pytest\n"),
      SonarQubePythonFile.create(generatedFile))
      .build();

    assertThat(context.isLikelyTestFile()).isTrue();
  }

  @Test
  void trailing_whitespace() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_trailing_whitespace.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var result = IpynbNotebookParser.parseNotebook(inputFile).get();
    var check = new TrailingWhitespaceCheck();
    var context = new PythonVisitorContext.Builder(
      TestPythonVisitorRunner.parseNotebookFile(result.locationMap(), result.contents()),
      SonarQubePythonFile.create(result))
      .build();
    check.scanFile(context);

    var issues = context.getIssues();
    assertThat(issues).hasSize(3);

    assertThat(issues.get(0).primaryLocation().startLine()).isEqualTo(14);
    assertThat(issues.get(0).primaryLocation().endLine()).isEqualTo(14);
    assertThat(issues.get(0).primaryLocation().startLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issues.get(0).primaryLocation().endLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);

    assertThat(issues.get(1).primaryLocation().startLine()).isEqualTo(17);
    assertThat(issues.get(1).primaryLocation().endLine()).isEqualTo(17);
    assertThat(issues.get(1).primaryLocation().startLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issues.get(1).primaryLocation().endLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);

    assertThat(issues.get(2).primaryLocation().startLine()).isEqualTo(20);
    assertThat(issues.get(2).primaryLocation().endLine()).isEqualTo(20);
    assertThat(issues.get(2).primaryLocation().startLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
    assertThat(issues.get(2).primaryLocation().endLineOffset()).isEqualTo(IssueLocation.UNDEFINED_OFFSET);
  }

  @Test
  void trailing_whitespace_compressed() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_trailing_whitespace_compressed.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var result = IpynbNotebookParser.parseNotebook(inputFile).get();
    var check = new TrailingWhitespaceCheck();
    var context = new PythonVisitorContext.Builder(
      TestPythonVisitorRunner.parseNotebookFile(result.locationMap(), result.contents()),
      SonarQubePythonFile.create(result))
      .build();
    check.scanFile(context);

    var issues = context.getIssues();
    assertThat(issues).hasSize(3);

    assertThat(issues.get(0).primaryLocation().startLine()).isEqualTo(1);
    assertThat(issues.get(0).primaryLocation().endLine()).isEqualTo(1);
    assertThat(issues.get(0).primaryLocation().startLineOffset()).isEqualTo(142);
    assertThat(issues.get(0).primaryLocation().endLineOffset()).isEqualTo(157); // Should be 154

    assertThat(issues.get(1).primaryLocation().startLine()).isEqualTo(1);
    assertThat(issues.get(1).primaryLocation().endLine()).isEqualTo(1);
    assertThat(issues.get(1).primaryLocation().startLineOffset()).isEqualTo(199);
    assertThat(issues.get(1).primaryLocation().endLineOffset()).isEqualTo(207); // Should be 204

    assertThat(issues.get(2).primaryLocation().startLine()).isEqualTo(1);
    assertThat(issues.get(2).primaryLocation().endLine()).isEqualTo(1);
    assertThat(issues.get(2).primaryLocation().startLineOffset()).isEqualTo(249);
    assertThat(issues.get(2).primaryLocation().endLineOffset()).isEqualTo(249); // Should be 255
  }

  @Test
  void multiline_string_in_source_array() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_multiline_string_in_array.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var result = IpynbNotebookParser.parseNotebook(inputFile).get();

    // Should not throw IllegalStateException("No IPythonLocation found for line ...")
    var fileInput = TestPythonVisitorRunner.parseNotebookFile(result.locationMap(), result.contents());

    var statements = fileInput.statements().statements();
    // Every statement lives on the same raw ipynb line, since the whole array element is one JSON string
    // spanning a single physical line; columns must still strictly increase to reflect their real position.
    assertThat(statements)
      .hasSize(8)
      .allSatisfy(stmt -> assertThat(stmt.firstToken().line()).isEqualTo(9))
      .extracting(stmt -> stmt.firstToken().column()).isSorted();
  }

  @Test
  void multiline_array_element_without_trailing_newline_continues_next_element() throws IOException {
    // "source": ["x = 1\ny", " = 2"]: the first element's last split line ("y") has no trailing newline of
    // its own, so the second element (" = 2") continues it on the same physical line instead of starting
    // a new one. This used to add one locationMap entry too many, misaligning every line after it.
    var inputFile = createInputFile(baseDir, "notebook_multiline_string_in_array_no_trailing_newline.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var result = IpynbNotebookParser.parseNotebook(inputFile).get();

    var fileInput = TestPythonVisitorRunner.parseNotebookFile(result.locationMap(), result.contents());

    var statements = fileInput.statements().statements();
    assertThat(statements).hasSize(2);
    assertThat(statements.get(0).firstToken().line()).isEqualTo(9);
    assertThat(statements.get(1).firstToken().line()).isEqualTo(9);
    // "y = 2" starts further along the raw ipynb line than "x = 1"
    assertThat(statements.get(1).firstToken().column()).isGreaterThan(statements.get(0).firstToken().column());
  }

  @Test
  void databricksSinglePercentCellMagicsShouldParseWithoutHidingPythonCells() throws IOException {
    var pretty = parseFixture("notebook_databricks_magics.ipynb");
    var compressed = parseFixture("notebook_databricks_magics_compressed.ipynb");

    var prettyTree = parseGeneratedFile(pretty);
    var compressedTree = parseGeneratedFile(compressed);

    var expectedKinds = List.of(
      Tree.Kind.ASSIGNMENT_STMT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.EXPRESSION_STMT,
      Tree.Kind.ASSIGNMENT_STMT,
      Tree.Kind.ASSIGNMENT_STMT,
      Tree.Kind.PRINT_STMT);
    assertThat(prettyTree.statements().statements()).extracting(Tree::getKind).containsExactlyElementsOf(expectedKinds);
    assertThat(compressedTree.statements().statements()).extracting(Tree::getKind).containsExactlyElementsOf(expectedKinds);

    // The parser consumes the original single-percent directive directly; no synthetic '%' is added.
    assertThat(prettyTree.statements().statements().subList(1, 5))
      .allSatisfy(cellMagic -> assertThat(cellMagic.firstToken().value()).isEqualTo("%"));

    // %time remains a line magic and does not hide the following Python assignment.
    var timeStatement = (ExpressionStatement) prettyTree.statements().statements().get(5);
    assertThat(timeStatement.expressions()).singleElement().extracting(Tree::getKind).isEqualTo(Tree.Kind.LINE_MAGIC);

    var prettyLastStatement = prettyTree.statements().statements().get(7);
    assertThat(prettyLastStatement.firstToken().line()).isEqualTo(55);
    assertThat(prettyLastStatement.firstToken().column()).isEqualTo(9);
    var compressedLastStatement = compressedTree.statements().statements().get(7);
    assertThat(compressedLastStatement.firstToken().line()).isEqualTo(1);
    assertThat(compressedLastStatement.firstToken().column()).isEqualTo(684);

    assertSymbolContinuityAcrossMagicCells(pretty, prettyTree, 3);
    assertSymbolContinuityAcrossMagicCells(compressed, compressedTree, 3);
  }

  @Test
  void databricksMultilineStringSourcesShouldPreservePythonBodyAndMappings() throws IOException {
    var file = parseFixture("notebook_databricks_string_sources.ipynb");

    var tree = parseGeneratedFile(file);

    assertThat(tree.statements().statements()).extracting(Tree::getKind).containsExactly(
      Tree.Kind.EXPRESSION_STMT,
      Tree.Kind.ASSIGNMENT_STMT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.ASSIGNMENT_STMT);
    var pythonDirective = (ExpressionStatement) tree.statements().statements().get(0);
    assertThat(pythonDirective.expressions()).singleElement().extracting(Tree::getKind).isEqualTo(Tree.Kind.LINE_MAGIC);
    assertThat(tree.statements().statements().get(3).firstToken().line()).isEqualTo(25);
    assertSymbolContinuityAcrossMagicCells(file, tree, 2);
  }

  @Test
  void databricksOpaqueBodiesShouldNotLeakIntoFollowingPythonCells() throws IOException {
    var file = parseFixture("notebook_databricks_opaque_bodies.ipynb");

    assertThat(file.contents()).isEqualTo("""
      %md
      It's ordinary Markdown
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      after_apostrophe = 1
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      %md
      unmatched (
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      after_parenthesis = after_apostrophe + 1
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      %%sql
      SELECT '( FROM values
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      after_double_percent = after_parenthesis + 1
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      %md-sandbox
      <div class="note">
      Markdown **is not** Python
      </div>
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      after_md_sandbox = after_double_percent + 1
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      %fs
      ls dbfs:/datasets
      rm -r dbfs:/tmp/non-python
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER
      after_fs = after_md_sandbox + 1
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER""");
    assertThat(file.parserConfiguration().opaqueCellRanges()).containsExactly(
      new OpaqueCellRange(1, 3),
      new OpaqueCellRange(6, 8),
      new OpaqueCellRange(11, 13),
      new OpaqueCellRange(16, 20),
      new OpaqueCellRange(23, 26));

    var tree = parseGeneratedFile(file);

    assertThat(tree.statements().statements()).extracting(Tree::getKind).containsExactly(
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.ASSIGNMENT_STMT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.ASSIGNMENT_STMT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.ASSIGNMENT_STMT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.ASSIGNMENT_STMT,
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.ASSIGNMENT_STMT);
    assertThat(tree.statements().statements().get(1).firstToken().line()).isEqualTo(15);
    assertThat(tree.statements().statements().get(3).firstToken().line()).isEqualTo(30);
    assertThat(tree.statements().statements().get(5).firstToken().line()).isEqualTo(45);
    assertThat(tree.statements().statements().get(7).firstToken().line()).isEqualTo(62);
    assertThat(tree.statements().statements().get(9).firstToken().line()).isEqualTo(78);
  }

  @Test
  void jupyterDoublePercentBodyShouldNotLeakIntoFollowingPythonCell() throws IOException {
    var file = parseFixture("notebook_jupyter_opaque_cell_magic.ipynb");

    assertThat(file.dialect()).isEqualTo(NotebookDialect.IPYTHON);
    assertThat(file.parserConfiguration().opaqueCellRanges()).containsExactly(new OpaqueCellRange(1, 3));

    var tree = parseGeneratedFile(file);

    assertThat(tree.statements().statements()).extracting(Tree::getKind).containsExactly(
      Tree.Kind.CELL_MAGIC_STATEMENT,
      Tree.Kind.ASSIGNMENT_STMT);
    assertThat(tree.statements().statements().get(1).firstToken().line()).isEqualTo(15);
  }

  @Test
  void unrecognizedOrNonDatabricksSinglePercentMagicShouldNotHideInvalidPython() {
    var jupyterSinglePercentSql = parseFixture("notebook_jupyter_single_percent_sql.ipynb");
    var databricksUnknownMagic = parseFixture("notebook_databricks_unknown_magic.ipynb");
    var databricksSqlAlchemy = parseFixture("notebook_databricks_sqlalchemy.ipynb");

    assertThatThrownBy(() -> parseGeneratedFile(jupyterSinglePercentSql))
      .isInstanceOf(RecognitionException.class);
    assertThatThrownBy(() -> parseGeneratedFile(databricksUnknownMagic))
      .isInstanceOf(RecognitionException.class);
    assertThatThrownBy(() -> parseGeneratedFile(databricksSqlAlchemy))
      .isInstanceOf(RecognitionException.class);
  }

  private GeneratedIPythonFile parseFixture(String name) {
    var inputFile = createInputFile(frontendBaseDir, name, InputFile.Status.CHANGED, InputFile.Type.MAIN);
    return IpynbNotebookParser.parseNotebook(inputFile).orElseThrow();
  }

  private static FileInput parseGeneratedFile(GeneratedIPythonFile file) throws IOException {
    var parser = PythonParser.createIPythonParser(file.parserConfiguration());
    return new IPythonTreeMaker(file.locationMap()).fileInput(parser.parse(file.contents()));
  }

  private static void assertSymbolContinuityAcrossMagicCells(GeneratedIPythonFile file, FileInput tree, int expectedNames) {
    var context = new PythonVisitorContext.Builder(tree, SonarQubePythonFile.create(file)).build();
    var sharedNames = new ArrayList<Name>();
    tree.accept(new BaseTreeVisitor() {
      @Override
      public void visitName(Name name) {
        if ("shared".equals(name.name())) {
          sharedNames.add(name);
        }
        super.visitName(name);
      }
    });

    // Keep the context alive through symbol construction and ensure all Python cells share one module scope.
    assertThat(context.rootTree()).isSameAs(tree);
    assertThat(sharedNames).hasSize(expectedNames).allSatisfy(name -> assertThat(name.symbol()).isNotNull());
    assertThat(sharedNames.stream().map(Name::symbol).distinct()).hasSize(1);
  }

}
