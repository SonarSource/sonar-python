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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.LinkedList;
import java.util.Enumeration;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.net.URL;
import java.io.*;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;

import org.sonar.api.utils.SonarException;
import org.sonar.api.utils.ZipUtils;
import org.sonar.api.resources.ProjectFileSystem;

public class PythonComplexityAnalyzer {

  private static final String PYTHON = "python";
  private static final String ARGS = "all -v ";
  private static final String PYGENIE_RESOURCE = "/org/sonar/plugins/python/pygenie/";
  private static final Logger LOGGER = LoggerFactory.getLogger(PythonComplexityAnalyzer.class);

  private String commandTemplate;
  private File workDir;
  private File fallbackPath;

  public PythonComplexityAnalyzer(ProjectFileSystem projectFileSystem) {
    // TODO: provide the option for using an external pygenie
    //
    // String configuredPath = "";
    // if(!configuredPath.equals("")){
    // pygeniePath = configuredPath;
    // }
      
    workDir = projectFileSystem.getSonarWorkingDirectory();
    fallbackPath = new File(workDir, "pygenie.py");
      
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
    return parseOutput(callPygenie(path));
  }

  protected final void extractPygenie(File targetFolder) {
    LOGGER.debug("Extracting pygenie to '{}'", targetFolder.getAbsolutePath());

    try {
      URL url = PythonComplexityAnalyzer.class.getResource(PYGENIE_RESOURCE);
      File pygeniePath = new File(url.getFile());
      if (pygeniePath.exists()) {
        // not packed; probably development environment
        for (File f : FileUtils.listFiles(pygeniePath, null, false)) {
          FileUtils.copyFileToDirectory(f, targetFolder);
        }
      } else {
        // packed; probably deployed
        File packagePath = new File(StringUtils.substringBefore(url.getFile(), "!").substring(5));
	
	ZipUtils.unzip(packagePath, targetFolder, new ZipUtils.ZipEntryFilter() {
	    public boolean accept(ZipEntry entry) {
		// this only works without the first '/'
		return entry.getName().startsWith(PYGENIE_RESOURCE.substring(1));
	    }
	});
      }
    } catch (IOException e) {
      throw new SonarException("Cannot extract pygenie to '" +
			       targetFolder.getAbsolutePath() + "'", e);
    }
  }


  private List<String> callPygenie(String path) {
    List<String> lines = new LinkedList<String>();
    String command = commandTemplate + " " + path;

    LOGGER.debug("Calling pygenie: '{}'", command);

    try {
      Process p = Runtime.getRuntime().exec(command);
      BufferedReader stdInput = new BufferedReader(new InputStreamReader(p.getInputStream()));
      String s = null;

      while ((s = stdInput.readLine()) != null) {
        lines.add(s);
      }
    } catch (IOException e) {
      throw new SonarException("Error calling pygenie", e);
    }

    return lines;
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
        if ( !entityType.equals("C")) { // = C means 'class scope'
          int count = Integer.parseInt(tokens[2]);
          String name = tokens[1];
          ComplexityStat stat = new ComplexityStat(name, count);
          if (entityType.equals("X")) { // X means 'module scope'
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
}
