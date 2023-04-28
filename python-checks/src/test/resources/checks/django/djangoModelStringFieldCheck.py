from django.db import models

class NullFieldsModel(models.Model):
    name = models.CharField(max_length=50, null=True) # Noncompliant {{Replace this "null=True" flag with "blank=True".}}
#                                          ^^^^^^^^^ 0
    desc = models.TextField(max_length=50, null=True) # Noncompliant

class NullBlankFieldsModel(models.Model):
    name = models.CharField(max_length=50, null=True, blank=True) # Noncompliant {{Remove this "null=True" flag.}}
    desc = models.TextField(max_length=50, null=True, blank=True) # Noncompliant

class BlankFieldsModel(models.Model):
    name = models.CharField(max_length=50, blank=True)
    desc = models.TextField(max_length=50, blank=True)


