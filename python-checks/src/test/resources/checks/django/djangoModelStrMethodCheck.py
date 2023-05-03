from django.db import models

class InvalidModel(models.Model): # Noncompliant {{Define a "__str__" method for this Django model.}}
#     ^^^^^^^^^^^^ 0
    name = models.CharField(max_length=100)


class ValidModel(models.Model):
    name = models.CharField(max_length=100)
    def __str__(self):
        return self.name

class ValidModelInherited(ValidModel):
    name = models.CharField(max_length=100)

class NotModel:
    name = models.CharField(max_length=100)

class UnknownModelInherited(UnknownModel):
    name = models.CharField(max_length=100)


class MyAbstractModel(models.Model):
    name = models.CharField(max_length=100)
    class Meta:
        abstract=True
