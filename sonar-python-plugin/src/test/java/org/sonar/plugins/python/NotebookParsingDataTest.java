/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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

import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.python.IPythonLocation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

class NotebookParsingDataTest {
  @Test
  void testAddDelimiterToSource() {
    var data = new NotebookParsingData(new StringBuilder().append("First line"), new LinkedHashMap<>(), 0);

    data.addDelimiterToSource("Test", 1, 2);
    assertThat(data).extracting(d -> d.getAggregatedSource().toString()).isEqualTo("First lineTest");
    assertThat(data).extracting(d -> d.getAggregatedSourceLine()).isEqualTo(1);
    assertThat(data).extracting(d -> d.getLocationMap().size()).isEqualTo(1);
    assertThat(data).extracting(d -> d.getLocationMap().get(1).line()).isEqualTo(1);
    assertThat(data).extracting(d -> d.getLocationMap().get(1).column()).isEqualTo(2);
  }

  @Test
  void testAddLineToSource() {
    var data = new NotebookParsingData(new StringBuilder().append("First line"), new LinkedHashMap<>(), 5);

    data.addLineToSource("Test", new IPythonLocation(1, 2));
    assertThat(data).extracting(d -> d.getAggregatedSource().toString()).isEqualTo("First lineTest");
    assertThat(data).extracting(d -> d.getAggregatedSourceLine()).isEqualTo(6);
    assertThat(data).extracting(d -> d.getLocationMap().size()).isEqualTo(1);
    assertThat(data).extracting(d -> d.getLocationMap().get(6).line()).isEqualTo(1);
    assertThat(data).extracting(d -> d.getLocationMap().get(6).column()).isEqualTo(2);
  }

  @Test
  void testCombineEmpty() {
    var data = NotebookParsingData.empty();
    var data2 = NotebookParsingData.empty();

    data.combine(data2);
    assertEquals(data, NotebookParsingData.empty());
  }

  @Test
  void testCombine() {
    var location1 = new LinkedHashMap<Integer, IPythonLocation>();
    location1.put(1, new IPythonLocation(0, 1));
    var data = new NotebookParsingData(new StringBuilder().append("a"), location1, 4);

    var location2 = new LinkedHashMap<Integer, IPythonLocation>();
    location2.put(3, new IPythonLocation(2, 1));
    var data2 = new NotebookParsingData(new StringBuilder().append("b"), location2, 3);

    data.combine(data2);
    assertThat(data).extracting(notebook -> notebook.getAggregatedSourceLine()).isEqualTo(3);
    assertThat(data).extracting(notebook -> notebook.getLocationMap().size()).isEqualTo(2);
    assertThat(data).extracting(notebook -> notebook.getAggregatedSource().toString()).isEqualTo("ab");
  }

  @Test
  void testRemoveTrailingExtraLineDoesNothing() {
    var data = new NotebookParsingData(new StringBuilder().append("First line"), new LinkedHashMap<>(), 5);
    assertThat(data).extracting(d -> d.getAggregatedSource().toString()).isEqualTo("First line");
    data.removeTrailingExtraLine();
    assertThat(data).extracting(d -> d.getAggregatedSource().toString()).isEqualTo("First line");

    var emptyLines = new NotebookParsingData(new StringBuilder(), new LinkedHashMap<>(), 5);
    assertThat(emptyLines).extracting(d -> d.getAggregatedSource().toString()).isEqualTo("");
    emptyLines.removeTrailingExtraLine();
    assertThat(emptyLines).extracting(d -> d.getAggregatedSource().toString()).isEqualTo("");
  }

  @Test
  void testRemoveTrailingExtraLine() {
    var data = new NotebookParsingData(new StringBuilder().append("First line\n"), new LinkedHashMap<>(), 5);
    assertThat(data).extracting(d -> d.getAggregatedSource().toString()).isEqualTo("First line\n");
    data.removeTrailingExtraLine();
    assertThat(data).extracting(d -> d.getAggregatedSource().toString()).isEqualTo("First line");
  }

  @Test
  void testEquals() {
    var empty = NotebookParsingData.empty();
    assertEquals(empty, empty);
    assertNotEquals(empty, "test");

    var data = new NotebookParsingData(new StringBuilder().append("Test"), Map.of(), 0);
    assertNotEquals(data, empty);
  }

  @Test
  void testHashcode() {
    var empty = NotebookParsingData.empty();
    assertEquals(empty.hashCode(), empty.hashCode());
    assertNotEquals(empty.hashCode(), "test".hashCode());
  }
}
