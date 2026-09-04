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

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.python.EscapeCharPositionInfo;
import org.sonar.python.IPythonLocation;

import static org.assertj.core.api.Assertions.assertThat;

class IPythonCellInputTransformerTest {

  @Test
  void removesTheExactCommonPrefixAndAdjustsLocations() {
    var data = NotebookParsingData.empty();
    data.addLineToSource(" \tfirst\n", new IPythonLocation(10, 20, List.of(new EscapeCharPositionInfo(1, 1)), true));
    data.addLineToSource(" \t  second\n", new IPythonLocation(11, 20, List.of(new EscapeCharPositionInfo(1, 1)), true));
    data.addLineToSource(" \t \n", new IPythonLocation(12, 20, List.of(new EscapeCharPositionInfo(1, 1)), true));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed.getAggregatedSource()).hasToString("first\n  second\n \t \n");
    assertThat(transformed.getLocationMap()).containsEntry(1, new IPythonLocation(10, 23, List.of(), true));
    assertThat(transformed.getLocationMap()).containsEntry(2, new IPythonLocation(11, 23, List.of(), true));
    assertThat(transformed.getLocationMap()).containsEntry(3,
      new IPythonLocation(12, 20, List.of(new EscapeCharPositionInfo(1, 1)), true));
  }

  @Test
  void matchesIPythonCommonSpaceIndentation() {
    var data = NotebookParsingData.empty();
    data.addLineToSource("     if True:\n", new IPythonLocation(1, 1));
    data.addLineToSource("        action()\n", new IPythonLocation(2, 1));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed.getAggregatedSource()).hasToString("if True:\n   action()\n");
    assertThat(transformed.getLocationMap()).containsEntry(1, new IPythonLocation(1, 6));
    assertThat(transformed.getLocationMap()).containsEntry(2, new IPythonLocation(2, 6));
  }

  @Test
  void matchesIPythonCommonTabIndentation() {
    var data = NotebookParsingData.empty();
    data.addLineToSource("\tif True:\n", new IPythonLocation(1, 1, List.of(new EscapeCharPositionInfo(0, 1))));
    data.addLineToSource("\t\taction()\n", new IPythonLocation(2, 1,
      List.of(new EscapeCharPositionInfo(0, 1), new EscapeCharPositionInfo(1, 1))));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed.getAggregatedSource()).hasToString("if True:\n\taction()\n");
    assertThat(transformed.getLocationMap()).containsEntry(1, new IPythonLocation(1, 3));
    assertThat(transformed.getLocationMap()).containsEntry(2,
      new IPythonLocation(2, 3, List.of(new EscapeCharPositionInfo(0, 1))));
  }

  @Test
  void doesNotUseTheFirstLineIndentationWhenItIsNotCommon() {
    var data = NotebookParsingData.empty();
    data.addLineToSource(" if condition:\n", new IPythonLocation(1, 1));
    data.addLineToSource("  nested()\n", new IPythonLocation(2, 1));
    data.addLineToSource("top_level()\n", new IPythonLocation(3, 1));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed).isSameAs(data);
    assertThat(transformed.getAggregatedSource()).hasToString(" if condition:\n  nested()\ntop_level()\n");
    assertThat(transformed.getLocationMap()).containsEntry(1, new IPythonLocation(1, 1));
    assertThat(transformed.getLocationMap()).containsEntry(2, new IPythonLocation(2, 1));
    assertThat(transformed.getLocationMap()).containsEntry(3, new IPythonLocation(3, 1));
  }

  @Test
  void indentedCommentDoesNotOverrideAnUnindentedCodeLine() {
    var data = NotebookParsingData.empty();
    data.addLineToSource("    # comment\n", new IPythonLocation(1, 1));
    data.addLineToSource("if True:\n", new IPythonLocation(2, 1));
    data.addLineToSource("    action()\n", new IPythonLocation(3, 1));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed).isSameAs(data);
    assertThat(transformed.getAggregatedSource()).hasToString("    # comment\nif True:\n    action()\n");
  }

  @Test
  void alignedCommentParticipatesInTheCommonIndentation() {
    var data = NotebookParsingData.empty();
    data.addLineToSource("    # comment\n", new IPythonLocation(1, 1));
    data.addLineToSource("    first()\n", new IPythonLocation(2, 1));
    data.addLineToSource("    second()\n", new IPythonLocation(3, 1));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed.getAggregatedSource()).hasToString("# comment\nfirst()\nsecond()\n");
  }

  @Test
  void doesNotTreatTabsAndSpacesAsEquivalentIndentation() {
    var data = NotebookParsingData.empty();
    data.addLineToSource("\tfirst\n", new IPythonLocation(1, 1));
    data.addLineToSource("    second\n", new IPythonLocation(2, 1));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed).isSameAs(data);
    assertThat(transformed.getAggregatedSource()).hasToString("\tfirst\n    second\n");
  }

  @Test
  void spacesAndTabsOnlyLinesDoNotLimitTheCommonIndentationAndArePreserved() {
    var data = NotebookParsingData.empty();
    data.addLineToSource("  first\n", new IPythonLocation(1, 1));
    data.addLineToSource(" \t \n", new IPythonLocation(2, 1));
    data.addLineToSource("  second\n", new IPythonLocation(3, 1));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed.getAggregatedSource()).hasToString("first\n \t \nsecond\n");
    assertThat(transformed.getLocationMap()).containsEntry(1, new IPythonLocation(1, 3));
    assertThat(transformed.getLocationMap()).containsEntry(2, new IPythonLocation(2, 1));
    assertThat(transformed.getLocationMap()).containsEntry(3, new IPythonLocation(3, 3));
  }

  @Test
  void pythonWhitespaceOnlyLinesDoNotLimitTheCommonIndentationAndArePreserved() {
    var data = NotebookParsingData.empty();
    data.addLineToSource("  first\n", new IPythonLocation(1, 1));
    data.addLineToSource("  \r\n", new IPythonLocation(2, 1));
    data.addLineToSource("\r\n", new IPythonLocation(3, 1));
    data.addLineToSource("  \f\n", new IPythonLocation(4, 1, List.of(new EscapeCharPositionInfo(2, 1))));
    data.addLineToSource("  second\n", new IPythonLocation(5, 1));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed.getAggregatedSource()).hasToString("first\n  \r\n\r\n  \f\nsecond\n");
    assertThat(transformed.getLocationMap()).containsEntry(2, new IPythonLocation(2, 1));
    assertThat(transformed.getLocationMap()).containsEntry(3, new IPythonLocation(3, 1));
    assertThat(transformed.getLocationMap()).containsEntry(4,
      new IPythonLocation(4, 1, List.of(new EscapeCharPositionInfo(2, 1))));
  }

  @Test
  void leadingWhitespaceOnlyLinesDoNotLimitTheCommonIndentation() {
    var data = NotebookParsingData.empty();
    data.addLineToSource("\r\n", new IPythonLocation(1, 1));
    data.addLineToSource("\f\n", new IPythonLocation(2, 1));
    data.addLineToSource("  first\n", new IPythonLocation(3, 1));
    data.addLineToSource("  second\n", new IPythonLocation(4, 1));

    var transformed = IPythonCellInputTransformer.removeCommonLeadingIndentation(data);

    assertThat(transformed.getAggregatedSource()).hasToString("\r\n\f\nfirst\nsecond\n");
    assertThat(transformed.getLocationMap()).containsEntry(3, new IPythonLocation(3, 3));
    assertThat(transformed.getLocationMap()).containsEntry(4, new IPythonLocation(4, 3));
  }
}
