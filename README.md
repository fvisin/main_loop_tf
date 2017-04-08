A main loop based on dataset loaders

## Usage
To use the main loop, simply create a file with this code and call it.

``` python
if __name__ == '__main__':
    import sys

    from TF_main_loop.main import run

    from model import build_model

    argv = sys.argv
    # Here you can potentially put some constant params you want to pass to the
    # main loop
    run(argv, build_model)
```


## How to add your own parameters
To add some model specific parameters to the list of parameters, just specify
them in your model file with the usual gflags syntax:

``` python
    import gflags

    gflags.DEFINE_integer('my_custom_param', 0, 'This sets up a custom param')

    def build_model():
        cfg = gflags.cfg
        print('This is your param value {}'.format(cfg.my_custom_param))
        pass
```

You can find a list of DEFINE* methods 
[here](https://github.com/google/python-gflags/blob/master/gflags/__init__.py)

You can also flag some options as required with:
```python
    gflags.mark_flag_as_required('required1')
    gflags.mark_flags_as_required(['required1', 'required2])
```

### How to add lists of lists
To add list of lists (e.g. [[10, 10], [20, 20]]) you can use the gflags
extensions in gflags_ext:

``` python
import gflags
import sys
from TF_main_loop import gflags_ext

gflags_ext.DEFINE_intlist('a', [[10, 10], [20, 20]], 'A list of ints')
```

See `gflags_ext` for the other DEFINEs.
