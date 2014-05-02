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
package org.sonar.python;

import com.google.common.base.Charsets;
import com.google.common.collect.ImmutableList;
import com.sonar.sslr.api.Grammar;
import org.junit.Test;
import org.sonar.python.api.PythonMetric;
import org.sonar.squidbridge.AstScanner;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.api.SourceProject;
import org.sonar.squidbridge.indexer.QueryByType;

import java.io.File;

import static org.fest.assertions.Assertions.assertThat;

public class PythonAstScannerTest {

  @Test
  public void files() {
    AstScanner<Grammar> scanner = PythonAstScanner.create(new PythonConfiguration(Charsets.UTF_8));
    scanner.scanFiles(ImmutableList.of(new File("src/test/resources/metrics/lines.py"), new File("src/test/resources/metrics/comments.py")));
    SourceProject project = (SourceProject) scanner.getIndex().search(new QueryByType(SourceProject.class)).iterator().next();
    assertThat(project.getInt(PythonMetric.FILES)).isEqualTo(2);
  }

  @Test
  public void comments() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/metrics/comments.py"));
    assertThat(file.getInt(PythonMetric.COMMENT_LINES)).isEqualTo(1);
    assertThat(file.getNoSonarTagLines()).contains(3).hasSize(1);
  }

  @Test
  public void lines() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/metrics/lines.py"));
    assertThat(file.getInt(PythonMetric.LINES)).isEqualTo(6);
  }

  @Test
  public void lines_of_code() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/metrics/lines_of_code.py"));
    assertThat(file.getInt(PythonMetric.LINES_OF_CODE)).isEqualTo(1);
  }

  @Test
  public void statements() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/metrics/statements.py"));
    assertThat(file.getInt(PythonMetric.STATEMENTS)).isEqualTo(1);
  }

  @Test
  public void functions() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/metrics/functions.py"));
    assertThat(file.getInt(PythonMetric.FUNCTIONS)).isEqualTo(1);
  }

  @Test
  public void classes() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/metrics/classes.py"));
    assertThat(file.getInt(PythonMetric.CLASSES)).isEqualTo(1);
  }

  @Test
  public void complexity() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/metrics/complexity.py"));
    assertThat(file.getInt(PythonMetric.COMPLEXITY)).isEqualTo(10);
  }

}
