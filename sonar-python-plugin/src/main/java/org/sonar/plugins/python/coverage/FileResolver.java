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
package org.sonar.plugins.python.coverage;

import org.sonar.api.resources.File;
import org.sonar.api.resources.Project;
import org.sonar.api.scan.filesystem.ModuleFileSystem;

public class FileResolver {

  private final Project project;
  private final ModuleFileSystem fs;

  public FileResolver(Project project, ModuleFileSystem fs) {
    this.project = project;
    this.fs = fs;
  }

  public File getFile(String path) {
    java.io.File file = new java.io.File(path);
    if (!file.isAbsolute()) {
      file = new java.io.File(fs.baseDir(), path);
    }
    return File.fromIOFile(file, project);
  }

}
