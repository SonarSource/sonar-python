/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;


@Rule(key = "S9999")
public class TypePrinterCheck extends PythonSubscriptionCheck {

  private static final Logger LOG = LoggerFactory.getLogger(TypePrinterCheck.class);
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      FileInput fileInput = (FileInput) ctx.syntaxNode();
      List<DataObject> data = new ArrayList<>();
      TypeVisitor typeVisitor = new TypeVisitor(data);
      fileInput.accept(typeVisitor);
      Gson gson = new GsonBuilder().setPrettyPrinting().create();
      String json = gson.toJson(data);
      LOG.info(json);
      try {
        File file = new File(ctx.workingDirectory(), "types_" + ctx.pythonFile().fileName() + ".json");
        FileWriter fileWriter = new FileWriter(file);
        fileWriter.write(json);
        fileWriter.close();

      } catch (IOException e) {
        e.printStackTrace();
      }
    });
  }
}

class TypeVisitor extends BaseTreeVisitor {

  List<DataObject> typeList;
  public TypeVisitor(List<DataObject> typeList) {
    this.typeList = typeList;
  }
  @Override
  public void visitName(Name name) {
    InferredType type = name.type();
    Symbol symbol = name.symbol();
    String symbolKind = "#no_symbol_kind";
    if (symbol != null) {
      symbolKind = symbol.kind().name();
    }
    String typeName = InferredTypes.typeName(type) != null ? InferredTypes.typeName(type) : "#no_type_name_available";
    DataObject dataObject = new DataObject(
      name.name(),
      name.firstToken().line(),
      name.firstToken().column(),
      name.lastToken().line(),
      name.lastToken().column() + name.lastToken().value().length(),
      name.getKind().name(),
      typeName,
      symbolKind);
    typeList.add(dataObject);
  }

  @Override
  public void visitNumericLiteral(NumericLiteral numericLiteral) {
    InferredType type = numericLiteral.type();
    String typeName = InferredTypes.typeName(type) != null ? InferredTypes.typeName(type) : "#no_type_name_available";
    DataObject dataObject = new DataObject(
      numericLiteral.valueAsString(),
      numericLiteral.firstToken().line(),
      numericLiteral.firstToken().column(),
      numericLiteral.lastToken().line(),
      numericLiteral.lastToken().column() + numericLiteral.lastToken().value().length(),
      numericLiteral.getKind().name(),
      typeName,
      "#no_symbol_kind");
    typeList.add(dataObject);
  }

  @Override
  public void visitStringLiteral(StringLiteral stringLiteral) {
    InferredType type = stringLiteral.type();
    String typeName = InferredTypes.typeName(type) != null ? InferredTypes.typeName(type) : "#no_type_name_available";
    DataObject dataObject = new DataObject(
      stringLiteral.trimmedQuotesValue(),
      stringLiteral.firstToken().line(),
      stringLiteral.firstToken().column(),
      stringLiteral.lastToken().line(),
      stringLiteral.lastToken().column() + stringLiteral.lastToken().value().length(),
      stringLiteral.getKind().name(),
      typeName,
      "#no_symbol_kind");
    typeList.add(dataObject);
  }

}

class DataObject {
  final String text;

  final int startLine;

  final int startColumn;

  final int endLine;

  final int endCol;

  final String syntaxRole;

  final String type;

  final String symbolKind;

  public DataObject(String text, int startLine, int startCol, int endLine, int endCol, String syntaxRole, String type, String symbolKind) {
    this.text = text;
    this.startLine = startLine;
    this.startColumn = startCol;
    this.endLine = endLine;
    this.endCol = endCol;
    this.syntaxRole = syntaxRole;
    this.type = type;
    this.symbolKind = symbolKind;
  }
}
