# Pytest

## Unit test vocabulary

* A unit test is a small piece of code: a method or function, a module or class, a small group of related classes.
* An automated unit test: designed by a human, runs without intervention, reports either pass or fail

Strictly speaking, a unit test do not use:
* the filesystem
* a database
* the network

Unit test code must run on memory.

Three parts of a test:
* arrange: set up the object to be tested, and collaborators
* act: exercise the unit under test
* assert: make claims about what happened

```python
import unittest

class PhoneBook():
    def __init__(self):
        self.numbers = {}

    def add(self, name, number):
        self.numbers[name] = number
    
    def lookup(self, name):
        return self.numbers[name]

class PhoneBookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.phonebook = PhoneBook()
    
    def tearDown(self) -> None:
        pass
    
    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    def setUpClass(cls):
        pass
    
    def test_lookup_by_name(self):
        self.phonebook.add("Bob", "12345")
        number = self.phonebook.lookup("Bob")
        self.assertEqual("12345", number)
    
    def test_missing_name(self):
        with self.assertRaises(KeyError):
            self.phonebook.lookup("missing")
    
    def test_empty_phonebook_is_consistent(self):
        self.assertTrue(self.phonebook.is_consistent())
    
    def test_is_consistent_with_different_entries(self):
        pass
    
    def test_inconsistent_with_duplicate_entries(self):
        pass
    
    def test_inconsistent_with_duplicate_prefix(self):
        pass
    
    def test_phonebook_adds_names_and_numbers(self):
        self.phonebook.add("Sue", "123343")
        self.assertIn("Sue", self.phonebook.get_names())
        self.assertIn("123343", self.phonebook.get_numbers())
```

Launch test
```shell
python -m unittest # in the same folder than the test code
```

Rules on Test Cases:
* should run independently
* sould not have side effect on other unit test
* multiple test cases can test the same piece of code

**Test Suite**: is a list of unit test that are executed together.
A test suite can be made of different test method from different TestCase classes.

## Skip test attribute/mark
```python
@unittest.skip("WIP")
def test_toto(self):
    pass
```

## Setup method
`setup()` is inherited from the `unittest.TestCase` class.
setUp is called before every test method.

## Teardown method
tearDown is called after every test method.

## Text Fixture
https://docs.python.org/fr/3.8/library/unittest.html#assert-methods

**setup/tearDown is an example of a test fixture.**

Order of execution:
* setUp() (if setup fail the following is not executing)
* TestCaseMethod() (multiple)
* tearDown() (always run)

## Test naming
Should give a good indication of what functionnality it tests. It should be enough specific.

## Why should we unit tests

Specify unit test make pin down what each unit of the system will do. It can expose problem early in the development process.
Writting unit test, helps defining what we try to achieve, and spot gaps in the understanding of the problem.

For example we can define a test method by business rule.

A unit test specifies the behaviour of the unit under test.

Enable design units, decompose into testable units. Loose coupling & high cohesion more likely to be more mantainable.

Design interface & implementation separately, by interface I mean function name, parameter list, return type.

Unit code helps detect regression, ensure previously developed and tested software still performs after a change.

Readable assertion messages, helps what's no longer working.

## Limitations of unit test
* hard to write if units have many dependencies
* test scenarios may not occur in production
* tests may note notice failures
* unit tests do not find integration or non-functional problems

## Test Driven Development
Write one test case for a small piece of functionality, write just enough implementation for the test to pass, and then refactor until the design is good. Then write more functionality.

Write One Test => Make it pass => Refactor => (again)

## Example of interface
```python
class TelemetryDiagnosticControls:
    DiagnosticChannelConnectionString = "*11s"

    def __init__(self, client=None):
        self.telemetry_client = client or TelemetryClient()
        self.diagnostic_info = ""
```

## pytest
pytest is a popular alternative to unittest. It is not a member of the xUnit family. It's not in the standard Python installation.

```python
class Phonebook:

    def __init__(self) -> None:
        self.numbers = {}
    
    def add(self, name, number):
        self.numbers[name] = number
    
    def lookup(self, name):
        return self.numbers[name]
    
    def names(self):
        return set(self.numbers.keys())

def test_lookup_by_name():
    phonebook = Phonebook()
    phonebook.add("Bob", "1234")
    assert "1234" == phonebook.lookup("Bob")

def test_phonebook_contains_all_names():
    phonebook = Phonebook()
    phonebook.add("Bob", "1234")
    assert phonebook.names() == {"Bob"}

def test_missing_name_raises_error():
    phonebook = Phonebook()
    with pytest.raises(KeyError):
        phonebook.lookup("Bob")
```

## Test Fixture with pytest
```python
@pytest.fixture
def phonebook():
    return Phonebook()
    
def test_lookup_by_name(phonebook):
    phonebook.add("Bob", "1234")
    assert "1234" == phonebook.loopup("Bob")
```

## Native test fixture with pytest
```python
class Phonebook:

    def __init__(self, cache_directory):
        self.numbers = {}
        self.filename = os.path.join(cache_directory, "phonebook.txt")
        self.cache = open(self.filename, "w")
    
    def clear(self):
        self.cache.close()
        os.remove(self.filename)

@pytest.fixture
def phonebook(tmpdir): # will call the tmpdir (native pytest fixture)
    """
    Provides empty phonebook
    """
    phonebook = Phonebook(tmpdir)
    yield phonebook
    phonebook.clear() # run after each test case
    
def test_lookup_by_name(phonebook):
    phonebook.add("Bob", "1234")
    assert "1234" == phonebook.lookup("Bob")
```

Get all pytest built-in fixtures
```shell
pytest --fixtures
```

Better to put shared fixtures in the `conftest.py` file. Consider by pytest as a local plugin'. Can contain hook function as well as fixtures. Any fixture here are available to all test module in this folder and sub folders.

## Using markers
```python
@python.mark.slow
def test_test():
    pass
```
```shell
pytest -m "not slow"
```

Add configuration in the vscode to test only one marker name.

```ini
[pytest]
addopts = --strict # only allow marker to be used that are listed in this file
markers =
    slow: run test ..
```

See all native markers
```shell
pytest --markers
```

```python
@pytest.mark.skip("WIP")
def test_wip():
    pass

@pytest.mark.skipuf(...)
def test_wip():
    pass
```

## Plugin to pytest
Example: present test result on a webpage: *pypi.org/project/pytest-html*
and then
```shell
pytest --html=report.html
```

## Test Doubles: Mocks, Stubs and Fakes

### Stub
```python
class StubSensor:
    def sample_pressure(self):
        return 15

def test_low_pressure_activates_alarm():
    alarm = Alarm(sensor=StubSensor())
    alarm.Check()
    assert alarm.is_alarm_on

# with unittest.Mock
def normal_pressure_alarm_stays_off():
    stub_sensor = Mock(sensor)
    stub_sensor.sample_pressure.return_value = 18
    alarm = Alarm(stub_sensor)
    alarm.check()
    assert not alarm.is_alarm_on
```

### Fakes
Looks good from the outside, has a real implementation with logic and behavior.

File: replace with StringIO, database: replace with in-memory database, webserver: replace with lightweight web server.

```python
class HtmlPagesConverter:
    def __init__(self, open_file):
        self.open_file = open_file
        self._find_page_breaks()
    
    ...

def test_convert_quotes();
    fake_file = io.StringIO("quote: ' ")
    converter = HtmlPagesConverter(open_file=fake_file)
    converted_next = converter.get_html_page(0)
    assert converted_text = "quote: ' "
```

### Dummy
A dummy is usually a `None` or empty array/list.
We use a dummy wher we are forced to pass an argument to a function as an argument of the test, but it is not used.

### Mock or Spy
Mock or Spy does all that a stub do.
* Stub: will not fail the test
* Mock or spy: can fail the test if it's not called correctly

Three kinds of assert:
* return value or exception
* state change: use an API to query the new state
* Method call: did a specific method get called on a collaborator

The third kind uses a mock or a spy.
A Spy records the method calls it receives, so you can assert they were correct.

```python
class MyService:
    def __init__(self, sso_registry):
        self.sso_registry = sso_registry
    
    def handle(self, request, sso_token):
        if self.sso_registry.is_valid(sso_token):
            return Response("Hello {0}!".format(request.name))
        else:
            return Response("Please sign in")

def test_single_sign_on():
    spy_sso_registry = Mock(SingleSignOnRegistry)
    service = MyService(spy_sso_registry)
    token = SSOToken()
    service.handle(Request("Emily"), token)
    spy_sso_registry.is_valid.assert_called_with(token) # here is the "spy" magic


def test_single_sign_on():
    spy_sso_registry = Mock(SingleSignOnRegistry)
    spy_sso_registry.is_valid_.return_value = False
    service = MyService(spy_sso_registry)
    token = SSOToken()
    response = service.handle(Request("Emily"), token)
    spy_sso_registry.is_valid.assert_called_with(token) # here is the "spy" magic
    assert response.text = "Please sign in"
```

### Mock
Looks as other test from the outstide, but expect certain method calls and with certain argument, otherwise raise an error.

```python
def confirm_token(correct_token):
    def is_valid(actual_token):
        if actual_token != correct_token:
            raise ValueError("wrong token received")
    
    return is_valid

def test_single_sign_on_receives_correct_token():
    mock_sso_registry = Mock(SingleSignOnRegistry)
    correct_token = SSOToken()
    mock_sso_registry.is_valid = Mock(side_effect=confirm_token(correct_token))
    service = MyService(mock_sso_registry)
    service.handle(Request("Emily"), correct_token)
    mock_sso_registry.is_valid.assert_called()
```

### Monkey patching
Monkey patching is another name for meta programing, dynamically change an attribute at run time.
Can be useful to insert a test double.

```python
from unittest.mock import patch, Mock
from alarm import Alarm

def test_alarm_with_high_pressure_value():
    with patch("alarm.Sensor") as test_sensor_class: # strange choice of patch "name" alarm.Sensor ..
        test_sensor_instance = Mock()
        test_sensor_instance.sample_pressure.return_value = 22
        test_sensor_class.return_value = test_sensor_instance

        alarm = Alarm()
        alarm.check()

        assert alarm.is_alarm_on

@patch("alarm.Sensor")
def test_alarm_with_too_loo_pressure_value(test_sensor_class):
    test_sensor_instance = Mock()
    test_sensor_instance.sample_pressure.return_value = 16
    test_sensor_class.return_value = test_sensor_instance

    alarm = Alarm()
    alarm.check()

    assert alarm.is_alarm_on
```

Monkey patching is usually not recommended, it's better to modify the constructor.
In other cases, however, it is still tough, like on context class (__init__, __enter__, __exit__)

## Misc
```python
def test(x):
    x.append(3)
    return x

def test1(x):
    x += " end of text"
    return x

def factorial_1(x):
    if x == 0:
        return 0
    if x == 1:
        return 1
    
    return x * factorial_1(x - 1)
```