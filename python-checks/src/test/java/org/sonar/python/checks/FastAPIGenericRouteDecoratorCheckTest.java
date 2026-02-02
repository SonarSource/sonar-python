/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.python.checks.quickfix.PythonQuickFixVerifier;
import org.sonar.python.checks.utils.PythonCheckVerifier;

class FastAPIGenericRouteDecoratorCheckTest {

  private static final FastAPIGenericRouteDecoratorCheck check = new FastAPIGenericRouteDecoratorCheck();

  @Test
  void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/fastAPIGenericRouteDecorator.py", check);
  }

  @ParameterizedTest
  @MethodSource("quickFixWithMessageTestCases")
  void quickFixWithMessage(String testName, String before, String after, String expectedMessage) {
    PythonQuickFixVerifier.verify(check, before, after);
    PythonQuickFixVerifier.verifyQuickFixMessages(check, before, expectedMessage);
  }

  @ParameterizedTest
  @MethodSource("quickFixTestCases")
  void quickFix(String testName, String before, String after) {
    PythonQuickFixVerifier.verify(check, before, after);
  }

  static Stream<Arguments> quickFixWithMessageTestCases() {
    return Stream.of(
      Arguments.of(
        "basic GET",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.route("/users", methods=["GET"])
          def get_users():
              return {"users": []}""",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.get("/users")
          def get_users():
              return {"users": []}""",
        "Replace with \"get\""),
      Arguments.of(
        "POST method",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.route("/users", methods=["POST"])
          def create_user():
              return {}""",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.post("/users")
          def create_user():
              return {}""",
        "Replace with \"post\""),
      Arguments.of(
        "router DELETE",
        """
          from fastapi import APIRouter
          router = APIRouter()
          @router.route("/items", methods=["DELETE"])
          def delete_item():
              return {}""",
        """
          from fastapi import APIRouter
          router = APIRouter()
          @router.delete("/items")
          def delete_item():
              return {}""",
        "Replace with \"delete\""),
      Arguments.of(
        "lowercase method",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.route("/lower", methods=["get"])
          def lower_method():
              return {}""",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.get("/lower")
          def lower_method():
              return {}""",
        "Replace with \"get\""));
  }

  static Stream<Arguments> quickFixTestCases() {
    return Stream.of(
      Arguments.of(
        "methods first",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.route(methods=["GET"], path="/users")
          def get_users():
              return {}""",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.get(path="/users")
          def get_users():
              return {}"""),
      Arguments.of(
        "with additional params",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.route("/users", methods=["GET"], response_model=None)
          def get_users():
              return {}""",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.get("/users", response_model=None)
          def get_users():
              return {}"""),
      Arguments.of(
        "methods middle position",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.route("/users", methods=["PUT"], status_code=200, response_model=None)
          def update_user():
              return {}""",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.put("/users", status_code=200, response_model=None)
          def update_user():
              return {}"""),
      Arguments.of(
        "only one arg",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.route(methods=["GET"])
          def get_users():
              return {}""",
        """
          from fastapi import FastAPI
          app = FastAPI()
          @app.get()
          def get_users():
              return {}"""));
  }
}
