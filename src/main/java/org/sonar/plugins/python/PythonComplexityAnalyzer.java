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

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.ZipEntry;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.sonar.api.resources.Project;
import org.sonar.api.utils.SonarException;
import org.sonar.api.utils.ZipUtils;

public class PythonComplexityAnalyzer {

  private static final String PYTHON = "python";
  private static final String ARGS = "all -v ";
  private static final String PYGENIE_DIR = "/org/sonar/plugins/python/pygenie/";
  private static final String PYGENIE_SCRIPT = "pygenie.py";

  private String commandTemplate;

  public PythonComplexityAnalyzer(Project project) {
    // TODO: provide the option for using an external pygenie
    //
    // String configuredPath = "";
    // if(!configuredPath.equals("")){
    // pygeniePath = configuredPath;
    // }

    File workDir = project.getFileSystem().getSonarWorkingDirectory();
    File fallbackPath = new File(workDir, PYGENIE_DIR + PYGENIE_SCRIPT);

    String pygeniePath = "";
    if ( !fallbackPath.exists()) {
      extractPygenie(workDir);
      if ( !fallbackPath.exists()) {
        throw new SonarException(fallbackPath.getAbsolutePath() + " cannot be found");
      }
    }
    pygeniePath = fallbackPath.getAbsolutePath();

    commandTemplate = PYTHON + " " + pygeniePath + " " + ARGS;
  }

  public List<ComplexityStat> analyzeComplexity(String path) {
    return parseOutput(Utils.callCommand(commandTemplate + " " + path, null));
  }

  protected final void extractPygenie(File targetFolder) {
    PythonPlugin.LOG.debug("Extracting pygenie to '{}'", targetFolder.getAbsolutePath());

    try {
      URL url = PythonComplexityAnalyzer.class.getResource(PYGENIE_DIR);
      File pygeniePath = new File(url.getFile());
      if (pygeniePath.exists()) {
        // not packed; probably development environment
        for (File f : FileUtils.listFiles(pygeniePath, null, false)) {
          FileUtils.copyFileToDirectory(f, new File(targetFolder, PYGENIE_DIR));
        }
      } else {
        // packed; probably deployed
        File packagePath = new File(StringUtils.substringBefore(url.getFile(), "!").substring(5));

        ZipUtils.unzip(packagePath, targetFolder, new PygenieOnlyFilter());
      }
    } catch (IOException e) {
      throw new SonarException("Cannot extract pygenie to '" + targetFolder.getAbsolutePath() + "'", e);
    }
  }

  private List<ComplexityStat> parseOutput(List<String> lines) {
    // Parse the output of pygenie. Example of the format:
    //
    // File: /home/wenns/src/test-projects/python/store.py
    // Type Name Complexity
    // -------------------------------------------------------------
    // M Store.setRating 3
    // M Store.setPredictedRating 3
    // ... ... ...

    List<ComplexityStat> stats = new LinkedList<ComplexityStat>();

    List<String> linesWithoutHeader = lines.subList(2, lines.size());

    ComplexityStat fileStat = null;
    for (String line : linesWithoutHeader) {
      line = line.trim();

      String[] tokens = line.split(" +");
      if (tokens.length == 3) {
        String entityType = tokens[0];
        if ( !"C".equals(entityType)) { // = C means 'class scope'
          int count = Integer.parseInt(tokens[2]);
          String name = tokens[1];
          ComplexityStat stat = new ComplexityStat(name, count);
          if ( "X".equals(entityType)) { // X means 'module scope'
            fileStat = stat;
          } else {
            stats.add(stat);
          }
        }
      }
    }
    stats.add(0, fileStat);

    return stats;
  }

  static class PygenieOnlyFilter implements ZipUtils.ZipEntryFilter {
    public boolean accept(ZipEntry entry) {
      // this only works without the first '/'
      return entry.getName().startsWith(PYGENIE_DIR.substring(1));
    }
  };
}
