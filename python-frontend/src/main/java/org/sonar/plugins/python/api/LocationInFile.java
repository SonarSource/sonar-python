/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
package org.sonar.plugins.python.api;

import java.util.Objects;
import org.sonar.python.types.protobuf.DescriptorsProtos;

public class LocationInFile {
  private final String fileId;
  private final int startLine;
  private final int startLineOffset;
  private final int endLine;
  private final int endLineOffset;

  public LocationInFile(String fileId, int startLine, int startLineOffset, int endLine, int endLineOffset) {
    this.fileId = fileId;
    this.startLine = startLine;
    this.startLineOffset = startLineOffset;
    this.endLine = endLine;
    this.endLineOffset = endLineOffset;
  }

  public LocationInFile(DescriptorsProtos.LocationInFile locationInFileProto) {
    fileId = locationInFileProto.getFileId();
    startLine = locationInFileProto.getStartLine();
    startLineOffset = locationInFileProto.getStartLineOffset();
    endLine = locationInFileProto.getEndLine();
    endLineOffset = locationInFileProto.getEndLineOffset();
  }

  public String fileId() {
    return fileId;
  }

  public int startLine() {
    return startLine;
  }

  public int startLineOffset() {
    return startLineOffset;
  }

  public int endLine() {
    return endLine;
  }

  public int endLineOffset() {
    return endLineOffset;
  }


  public DescriptorsProtos.LocationInFile toProtobuf() {
    return DescriptorsProtos.LocationInFile.newBuilder()
      .setFileId(fileId)
      .setStartLine(startLine)
      .setStartLineOffset(startLineOffset)
      .setEndLine(endLine)
      .setEndLineOffset(endLineOffset)
      .build();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    LocationInFile that = (LocationInFile) o;
    return startLine == that.startLine && startLineOffset == that.startLineOffset && endLine == that.endLine && endLineOffset == that.endLineOffset && fileId.equals(that.fileId);
  }

  @Override
  public int hashCode() {
    return Objects.hash(fileId, startLine, startLineOffset, endLine, endLineOffset);
  }
}
