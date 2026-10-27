from typing import Protocol, overload


# Biopython BasicChromosome.draw: preserve the known direct contract even when
# a distant third-party ancestor is unresolved.
class Component(external.Widget):
    def draw(self): ...


class Chromosome(Component):
    def draw(self, drawing): ...  # Noncompliant {{Remove parameter drawing or provide default value.}}


# Protocol ancestry must not hide a contract defined locally by the parent.
class Timetable(Protocol):
    def infer_manual_data_interval(self, *, run_after): ...


class WorkdayTimetable(Timetable):
    def infer_manual_data_interval(self): ...  # Noncompliant


# Airflow's example widens the accepted argument modes without changing the contract.
class WidenedWorkdayTimetable(Timetable):
    def infer_manual_data_interval(self, run_after): ...


# Source overloads retain their public contracts after the implementation.
class Overloaded:
    @overload
    def run(self, value: int): ...
    @overload
    def run(self, value: str): ...
    def run(self, value): ...


class OverloadedChild(Overloaded):
    def run(self): ...  # Noncompliant {{Add missing parameters value.}}


# Twisted Protocol.connectionLost: resolving the inherited default exposes a
# real narrowing of the contract.
class Connection:
    def connectionLost(self, reason=None): ...


class Client(Connection):
    def connectionLost(self, reason): ...  # Noncompliant {{Add a default value to parameter reason.}}


# Twisted udp.Port.writeSequence: seq is a positional rename, not an extra
# parameter. The additional required addr parameter is a genuine finding.
class Writer:
    def writeSequence(self, iovec): ...


class DatagramWriter(Writer):
    def writeSequence(self, seq, addr): ...  # Noncompliant {{Remove parameter addr or provide default value.}}
#                                ^^^^


# Twisted SSHClientTransport.sendDisconnect renames both parameters without
# changing their positional meaning: code is the reason code, reason the text.
class Transport:
    def sendDisconnect(self, reason, desc): ...


class ClientTransport(Transport):
    # FP: the shared name is mistaken for a moved parameter.
    def sendDisconnect(self, code, reason): ...  # Noncompliant {{Move parameter reason to position 1.}}


# pandas Series.drop_duplicates: adding optional keyword-only parameters keeps
# every call accepted by the parent's signature valid.
class IndexOperations:
    def drop_duplicates(self, *, keep="first"): ...


class Series(IndexOperations):
    def drop_duplicates(self, *, keep="first", inplace=False, ignore_index=False): ...
