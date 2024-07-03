/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.file.Path;
import java.util.Map;
import javax.annotation.CheckForNull;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.TextPointer;
import org.sonar.api.batch.fs.TextRange;

public class GeneratedIPythonFile implements InputFile {

  InputFile originalFile;
  String pythonContent;
  Map<Integer, Offset> offsetMap;

  public GeneratedIPythonFile(InputFile originalFile, String pythonContent, Map<Integer, Offset> locationMap) {
    this.originalFile = originalFile;
    this.pythonContent = pythonContent;
    this.offsetMap = locationMap;
  }

  public Map<Integer, Offset> offsetMap() {
    return offsetMap;
  }

  public record Offset(int line, int column) {
  }

  @Override
  public String relativePath() {
    return originalFile.relativePath();
  }

  @Override
  public String absolutePath() {
    return originalFile.absolutePath();
  }

  @Override
  public File file() {
    return originalFile.file();
  }

  @Override
  public Path path() {
    return originalFile.path();
  }

  @Override
  public URI uri() {
    return originalFile.uri();
  }

  @Override
  public String filename() {
    return originalFile.filename();
  }

  @CheckForNull
  @Override
  public String language() {
    return originalFile.language();
  }

  @Override
  public Type type() {
    return originalFile.type();
  }

  @Override
  public InputStream inputStream() throws IOException {
    return originalFile.inputStream();
  }

  @Override
  public String contents() throws IOException {
    return pythonContent;
  }

  @Override
  public Status status() {
    return originalFile.status();
  }

  @Override
  public int lines() {
    return originalFile.lines();
  }

  @Override
  public boolean isEmpty() {
    return originalFile.isEmpty();
  }

  @Override
  public TextPointer newPointer(int i, int i1) {
    Offset offset = offsetMap.get(i);
    return originalFile.newPointer(offset.line, i1 + offset.column);
  }

  @Override
  public TextRange newRange(TextPointer textPointer, TextPointer textPointer1) {
    throw new RuntimeException("Probably incorrect to use this");
  }

  @Override
  public TextRange newRange(int i, int i1, int i2, int i3) {
    Offset offsetFrom = offsetMap.get(i);
    Offset offsetTo = offsetMap.get(i2);
    return originalFile.newRange(offsetFrom.line(), i1 + offsetFrom.column(), offsetTo.line(), i3 + offsetTo.column());
  }

  @Override
  public TextRange selectLine(int i) {
    Offset offset = offsetMap.get(i);
    return originalFile.selectLine(offset.line());
  }

  @Override
  public Charset charset() {
    return originalFile.charset();
  }

  @Override
  public String md5Hash() {
    return originalFile.md5Hash();
  }

  @Override
  public String key() {
    return originalFile.key();
  }

  @Override
  public boolean isFile() {
    return originalFile.isFile();
  }
}
