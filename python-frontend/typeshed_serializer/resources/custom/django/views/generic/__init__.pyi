import django.views.generic.base as base
from django.views.generic.base import RedirectView as RedirectView
from django.views.generic.base import TemplateView as TemplateView
from django.views.generic.base import View as View
from django.views.generic.dates import ArchiveIndexView as ArchiveIndexView
from django.views.generic.dates import DateDetailView as DateDetailView
from django.views.generic.dates import DayArchiveView as DayArchiveView
from django.views.generic.dates import MonthArchiveView as MonthArchiveView
from django.views.generic.dates import TodayArchiveView as TodayArchiveView
from django.views.generic.dates import WeekArchiveView as WeekArchiveView
from django.views.generic.dates import YearArchiveView as YearArchiveView
from django.views.generic.detail import DetailView as DetailView
from django.views.generic.edit import CreateView as CreateView
from django.views.generic.edit import DeleteView as DeleteView
from django.views.generic.edit import FormView as FormView
from django.views.generic.edit import UpdateView as UpdateView
from django.views.generic.list import ListView as ListView

class GenericViewError(Exception): ...
