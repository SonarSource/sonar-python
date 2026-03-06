from SonarPythonAnalyzerFakeStub import CustomStubBase
from django.db.models.base import Model
from django.db.models.manager import Manager
from typing import Any

class UserManager(Manager):
    def create(self, **kwargs: Any) -> "AbstractBaseUser": ...
    def create_user(
        self,
        username: str,
        email: str | None = None,
        password: str | None = None,
        **extra_fields: Any,
    ) -> "AbstractBaseUser": ...
    def create_superuser(
        self,
        username: str,
        email: str | None = None,
        password: str | None = None,
        **extra_fields: Any,
    ) -> "AbstractBaseUser": ...

class AbstractBaseUser(Model, CustomStubBase):
    password: str
    def set_password(self, raw_password: str) -> None: ...
    def check_password(self, raw_password: str) -> bool: ...

class AbstractUser(AbstractBaseUser):
    username: str
    email: str
    objects: UserManager

class User(AbstractUser):
    objects: UserManager

class AnonymousUser(CustomStubBase): ...
