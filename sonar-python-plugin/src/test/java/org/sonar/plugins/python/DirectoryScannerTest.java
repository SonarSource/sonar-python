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
package org.sonar.plugins.python;

import org.junit.Test;
import org.sonar.api.utils.WildcardPattern;

import java.io.File;
import java.util.List;

import static org.fest.assertions.Assertions.assertThat;

public class DirectoryScannerTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/scanner").getAbsoluteFile();

  @Test
  public void noMatchedFile() throws Exception {
    assertThat(scan("dir/xxx")).isEmpty();
  }

  @Test
  public void simpleFile() throws Exception {
    assertThat(scan("dir/f1.txt")).containsOnly(new File(baseDir, "dir/f1.txt"));
  }

  @Test
  public void wildCard() throws Exception {
    assertThat(scan("*/f1.txt")).containsOnly(new File(baseDir, "dir/f1.txt"));
    assertThat(scan("**/f1.txt")).containsOnly(new File(baseDir, "dir/f1.txt"), new File(baseDir, "dir/subdir/f1.txt"));
  }

  private List<File> scan(String pattern) {
    DirectoryScanner scanner = new DirectoryScanner(baseDir, WildcardPattern.create(pattern));
    return scanner.getIncludedFiles();
  }

}
