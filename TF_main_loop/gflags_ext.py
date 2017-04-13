from gflags.argument_parser import BaseListParser, ArgumentSerializer
from gflags import DEFINE, DEFINE_flag, MultiFlag, FLAGS, Flag


class DictMultiFlag(MultiFlag):
    def parse(self, arguments):
        """Parses one or more arguments with the installed parser.
        Args:
            arguments: a single argument or a dict of arguments (typically a
            dict of default values); a single argument is converted
            internally into a dict containing one item.
        """
        if arguments is None:
            return {}

        if not isinstance(arguments, list):
            # Default value may be a list of values.  Most other arguments
            # will not be, so convert them into a single-item list to make
            # processing simpler below.
            arguments = [arguments]

        if self.present:
            # keep a backup reference to list of previously supplied option
            # values
            values = self.value
        else:
            # "erase" the defaults with an empty list
            values = {}

        for item in arguments:
            # have Flag superclass parse argument, overwriting
            # self.value reference
            Flag.Parse(self, item)  # also increments self.present
            values.update(self.value)

        # put list of option values back in the 'value' attribute
        self.value = values


class DictParser(BaseListParser):
    """Parser for a comma or white space-separated list or tuple of strings.

    This parser will return a list of lists of numbers or a list of
    numbers, depending on the input. The number will be converted in int
    or float depending on the out_type.
    """

    def __init__(self):
        BaseListParser.__init__(self, ',', 'comma')

    def parse(self, argument):
        """Override to support full CSV syntax."""
        if isinstance(argument, (tuple, list)):
            raise NotImplementedError()
        elif isinstance(argument, dict):
            return argument
        elif argument is None:
            return argument
        else:
            # Make all brakets square brakets
            argument = argument.split('=')
            return {argument[0]: eval(argument[1])}


class ListOfListParser(BaseListParser):
    """Parser for a comma or white space-separated list or tuple of strings.

    This parser will return a list of lists of numbers or a list of
    numbers, depending on the input. The number will be converted in int
    or float depending on the out_type.
    """

    def __init__(self, out_type=int):
        self.out_type = out_type
        BaseListParser.__init__(self, ',', 'comma')

    def parse(self, argument):
        """Override to support full CSV syntax."""
        if isinstance(argument, list):
            return argument
        elif argument is None:
            return []
        else:
            # Make all brakets square brakets
            argument = argument.replace('(', '[').replace(')', ']')
            argument = argument.replace(' ', '')  # remove spaces
            argument = argument.replace('[', '').split('],')
            if len(argument) == 1:
                argument = argument[0].replace(']', '').split(',')
                return [self.out_type(s) for s in argument]
            else:
                return [map(self.out_type, s.replace(']', '').split(','))
                        for s in argument]


def DEFINE_multidict(name, default, help, flag_values=FLAGS, **args):
    """Parses a dictionary

    Args:
      name: A string, the flag name.
      default: The default value of the flag.
      help: A help string.
      flag_values: FlagValues object with which the flag will be registered.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = DictParser()
    serializer = ArgumentSerializer()

    DEFINE_flag(DictMultiFlag(parser, serializer, name, default, help, **args),
                flag_values, module_name=None)


def DEFINE_intlist(name, default, help, flag_values=FLAGS, **args):
    """Parses a list of lists of ints

    Args:
      name: A string, the flag name.
      default: The default value of the flag.
      help: A help string.
      flag_values: FlagValues object with which the flag will be registered.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = ListOfListParser(int)
    DEFINE(parser, name, default, help, flag_values, None, **args)


def DEFINE_floatlist(name, default, help, flag_values=FLAGS, **args):
    """Parses a list of lists of floats

    The flag value is parsed with a CSV parser.

    Args:
      name: A string, the flag name.
      default: The default value of the flag.
      help: A help string.
      flag_values: FlagValues object with which the flag will be registered.
      **args: Dictionary with extra keyword args that are passed to the
          Flag __init__.
    """
    parser = ListOfListParser(float)
    DEFINE(parser, name, default, help, flag_values, None, **args)
