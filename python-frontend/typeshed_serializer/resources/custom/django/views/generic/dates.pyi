import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from django.core.paginator import Paginator
from django.db.models import Model
from django.http.request import HttpRequest
from django.http.response import HttpResponse
from django.views.generic.base import View
from django.views.generic.detail import (
    BaseDetailView,
    SingleObjectTemplateResponseMixin,
)
from django.views.generic.list import (
    MultipleObjectMixin,
    MultipleObjectTemplateResponseMixin,
)

class YearMixin:
    year_format: str
    year: Optional[str]
    def get_year_format(self) -> str: ...
    def get_year(self) -> str: ...
    def get_next_year(self, date: datetime.date) -> Optional[datetime.date]: ...
    def get_previous_year(self, date: datetime.date) -> Optional[datetime.date]: ...

class MonthMixin:
    month_format: str
    month: Optional[str]
    def get_month_format(self) -> str: ...
    def get_month(self) -> str: ...
    def get_next_month(self, date: datetime.date) -> Optional[datetime.date]: ...
    def get_previous_month(self, date: datetime.date) -> Optional[datetime.date]: ...

class DayMixin:
    day_format: str
    day: Optional[str]
    def get_day_format(self) -> str: ...
    def get_day(self) -> str: ...
    def get_next_day(self, date: datetime.date) -> Optional[datetime.date]: ...
    def get_previous_day(self, date: datetime.date) -> Optional[datetime.date]: ...

class WeekMixin:
    week_format: str
    week: Optional[str]
    def get_week_format(self) -> str: ...
    def get_week(self) -> str: ...
    def get_next_week(self, date: datetime.date) -> Optional[datetime.date]: ...
    def get_previous_week(self, date: datetime.date) -> Optional[datetime.date]: ...

class DateMixin:
    date_field: Optional[str]
    allow_future: bool
    uses_datetime_field: bool
    def get_date_field(self) -> str: ...
    def get_allow_future(self) -> bool: ...

class BaseDateListView(MultipleObjectMixin, DateMixin, View):
    allow_empty: bool
    date_list_period: str
    date_list: Any
    def get(self, request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse: ...
    def get_dated_items(self) -> Tuple[Any, Any, Dict[str, Any]]: ...
    def get_ordering(self) -> Optional[Any]: ...
    def get_dated_queryset(self, **lookup: Any) -> Any: ...
    def get_date_list_period(self) -> str: ...
    def get_date_list(
        self, queryset: Any, date_type: Optional[str] = ..., ordering: str = ...
    ) -> Any: ...

class BaseArchiveIndexView(BaseDateListView):
    context_object_name: str
    def get_dated_items(self) -> Tuple[Any, Any, Dict[str, Any]]: ...

class ArchiveIndexView(MultipleObjectTemplateResponseMixin, BaseArchiveIndexView):
    template_name_suffix: str

class BaseYearArchiveView(YearMixin, BaseDateListView):
    date_list_period: str
    make_object_list: bool
    def get_dated_items(self) -> Tuple[Any, Any, Dict[str, Any]]: ...
    def get_make_object_list(self) -> bool: ...

class YearArchiveView(MultipleObjectTemplateResponseMixin, BaseYearArchiveView):
    template_name_suffix: str

class BaseMonthArchiveView(YearMixin, MonthMixin, BaseDateListView):
    date_list_period: str
    def get_dated_items(self) -> Tuple[Any, Any, Dict[str, Any]]: ...

class MonthArchiveView(MultipleObjectTemplateResponseMixin, BaseMonthArchiveView):
    template_name_suffix: str

class BaseWeekArchiveView(YearMixin, WeekMixin, BaseDateListView):
    def get_dated_items(self) -> Tuple[Any, Any, Dict[str, Any]]: ...

class WeekArchiveView(MultipleObjectTemplateResponseMixin, BaseWeekArchiveView):
    template_name_suffix: str

class BaseDayArchiveView(YearMixin, MonthMixin, DayMixin, BaseDateListView):
    def get_dated_items(self) -> Tuple[Any, Any, Dict[str, Any]]: ...

class DayArchiveView(MultipleObjectTemplateResponseMixin, BaseDayArchiveView):
    template_name_suffix: str

class BaseTodayArchiveView(BaseDayArchiveView):
    def get_dated_items(self) -> Tuple[Any, Any, Dict[str, Any]]: ...

class TodayArchiveView(MultipleObjectTemplateResponseMixin, BaseTodayArchiveView):
    template_name_suffix: str

class BaseDateDetailView(YearMixin, MonthMixin, DayMixin, DateMixin, BaseDetailView):
    def get_object(self, queryset: Optional[Any] = ...) -> Any: ...

class DateDetailView(SingleObjectTemplateResponseMixin, BaseDateDetailView):
    template_name_suffix: str
