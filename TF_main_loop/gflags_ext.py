from gflags.argument_parser import BaseListParser
from gflags import DEFINE, FLAGS


class ListOfListParser(BaseListParser):
    """Parser for a comma-separated list of strings."""

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
            return [map(self.out_type, s.replace(']', '').split(','))
                    for s in argument]


def DEFINE_intlist(name, default, help, flag_values=FLAGS, **args):
    """Registers a flag whose value is a comma-separated list of strings.

    The flag value is parsed with a CSV parser.

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
    """Registers a flag whose value is a comma-separated list of strings.

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
