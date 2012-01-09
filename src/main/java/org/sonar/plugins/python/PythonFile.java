/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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

import java.util.List;

import org.sonar.api.resources.DefaultProjectFileSystem;
import org.sonar.api.resources.Directory;
import org.sonar.api.resources.File;
import org.sonar.api.resources.Language;

public class PythonFile extends File {

  private static final String INITFILE = "__init__.py";

  private boolean partOfPackage = false;
  private Directory parentPackage;
  private String packageKey;

  public PythonFile(String key) {
    this(key, null, false);
  }

  public PythonFile(String key, String packageKey, boolean partOfPackage) {
    super(key);
    this.partOfPackage = partOfPackage;
    this.packageKey = packageKey;
  }

  /** Creates a File from an io.file and a list of sources directories */
  public static File fromIOFile(java.io.File file, List<java.io.File> sourceDirs) {
    String relativePath = DefaultProjectFileSystem.getRelativePath(file, sourceDirs);
    if (relativePath != null) {
      java.io.File packageInitFile = new java.io.File(file.getParent(), INITFILE);
      return new PythonFile(relativePath, Directory.parseKey(new java.io.File(relativePath).getParent()), packageInitFile.isFile());
    }
    return null;
  }

  public Directory getParent() {
    if (partOfPackage) {
      if (parentPackage == null) {
        parentPackage = new PythonPackage(packageKey);
      }
      return parentPackage;
    }
    return super.getParent();
  }

  public Language getLanguage() {
    return Python.INSTANCE;
  }
}
