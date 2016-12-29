/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.plugins.python.coverage;

import java.io.File;
import java.util.Map;
import java.util.Optional;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpressionException;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.coverage.NewCoverage;
import org.sonar.plugins.python.EmptyReportException;
import org.sonar.plugins.python.PythonReportException;
import org.sonar.plugins.python.XmlReportParser;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXParseException;

public class CoberturaParser extends XmlReportParser {

  private static final Logger LOG = LoggerFactory.getLogger(CoberturaParser.class);

  /**
   * @throws PythonReportException
   */
  public void parseReport(File xmlFile, SensorContext context, final Map<InputFile, NewCoverage> coverageData) {
    LOG.info("Parsing report '{}'", xmlFile);

    try {
      Document xmlDocument = getDocumentBuilder().parse(xmlFile);

      collectPackageMeasures(xmlDocument, context, coverageData);

    } catch (SAXParseException e) {
      if (e.getLineNumber() == 1 && "Premature end of file.".equals(e.getMessage())) {
        LOG.debug("Unexpected end of file is encountered");
        throw new EmptyReportException();
      } else {
        throw new PythonReportException(e);
      }
    } catch (Exception e) {
      throw new PythonReportException(e);
    }
  }

  private void collectPackageMeasures(Document xmlDocument, SensorContext context, Map<InputFile, NewCoverage> coverageData) throws XPathExpressionException {
    NodeList classNodes = (NodeList) getXpath().compile("//package/classes/class").evaluate(xmlDocument, XPathConstants.NODESET);
    for (int i = 0; i < classNodes.getLength(); i++) {
      Element classNode = (Element) classNodes.item(i);
      collectFileMeasures(classNode, context, coverageData);
    }
  }

  private void collectFileMeasures(Element classNode, SensorContext context, Map<InputFile, NewCoverage> coverageData) throws XPathExpressionException {
    Optional<String> fileName = getStringAttribute(classNode, "filename");
    if (fileName.isPresent()) {
      InputFile inputFile = context.fileSystem().inputFile(context.fileSystem().predicates().hasPath(fileName.get()));
      if (inputFile != null) {
        NewCoverage coverage = coverageData.get(inputFile);
        if (coverage == null) {
          coverage = context.newCoverage().onFile(inputFile);
          coverageData.put(inputFile, coverage);
        }
        collectFileData(classNode, coverage);
      } else {
        LOG.debug("Cannot find file '{}', ignoring coverage measure", fileName);
      }
    } else {
      LOG.debug("No value for 'filename' in the measure, ignoring coverage measure");
    }
  }

  private void collectFileData(Element classNode, NewCoverage coverage) throws XPathExpressionException {
    NodeList lineNodes = classNode.getElementsByTagName("line");
    for (int i = 0; i < lineNodes.getLength(); i++) {
      Element lineNode = (Element) lineNodes.item(i);
      Optional<Integer> lineNumber = getIntegerAttribute(lineNode, "number");
      Optional<Integer> lineHits = getIntegerAttribute(lineNode, "hits");
      if (lineNumber.isPresent() && lineHits.isPresent()) {
        coverage.lineHits(lineNumber.get(), lineHits.get());
        boolean isBranch = getBooleanAttribute(lineNode, "branch");
        Optional<String> conditionCoverage = getStringAttribute(lineNode, "condition-coverage");
        if (isBranch && conditionCoverage.isPresent() && StringUtils.isNotBlank(conditionCoverage.get())) {
          String[] conditions = StringUtils.split(StringUtils.substringBetween(conditionCoverage.get(), "(", ")"), "/");
          coverage.conditions(lineNumber.get(), Integer.parseInt(conditions[1]), Integer.parseInt(conditions[0]));
        }
      } else {
        LOG.debug("lineId (={}) and/or lineHits (={}) not provided, ignoring line mesaure", lineNumber, lineHits);
      }
    }
  }

}
