from gflags.argument_parser import BaseListParser
from gflags import DEFINE, FLAGS


class ListOfListParser(BaseListParser):
    """Parser for a comma or white space-separated list of strings.

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
        elif not argument:
            return []
        else:
            argument = argument.replace(' ', '')  # remove spaces
            argument = argument.replace('[', '').split('],')
            if len(argument) == 1:
                argument = argument[0].replace(']', '').split(',')
                return [self.out_type(s) for s in argument]
            else:
                return [map(self.out_type, s.replace(']', '').split(','))
                        for s in argument]


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
