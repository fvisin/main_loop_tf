A main loop to use Tensorflow with custom data not in the protobuf format.
This code is optimized to work coupled with the 
[dataset loaders](https://github.com/fvisin/dataset_loaders/), a framework to 
load and preprocess your data with parallel threads.

### Attribution
If you use this code, please cite:
* \[1\] Francesco Visin - Main loop TF: a main loop for Tensorflow and custom data ([BibTeX](
        https://gist.github.com/fvisin/a77ca03074242cf47a1641a85ded885c#file-main_loop_tf-bib))

If you use the Dataset loaders, please also cite:
* \[1\] Francesco Visin, Adriana Romero - Dataset loaders: a python library to
    load and preprocess datasets ([BibTeX](
        https://gist.github.com/fvisin/7104500ae8b33c3b65798d5d2707ce6c#file-dataset_loaders-bib))

### Usage
To use the main loop, simply create a file with this code and call it.

``` python
if __name__ == '__main__':
    import sys

    from main_loop_tf.main import run

    from model import build_model

    argv = sys.argv
    # Here you can potentially put some constant params you want to pass to the
    # main loop
    run(argv, build_model)
```


### How to add your own parameters
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

#### How to add lists of lists
To add list of lists (e.g. [[10, 10], [20, 20]]) you can use the gflags
extensions in gflags_ext:

``` python
import gflags
import sys
from main_loop_tf import gflags_ext

gflags_ext.DEFINE_intlist('a', [[10, 10], [20, 20]], 'A list of ints')
```

See `gflags_ext` for the other DEFINEs.

#### Paths
The models will be **saved** in:
  `<checkpoints_basedir>(/<suite_name>)/<model_name>(_model_suffix)`
  
and **restored** from:
  `<checkpoints_basedir>(/<restore_suite>)/<restore_model>`
  
* *model_name* and *restore_model*: default to the hash of the hyperparameters if not specified
* *suite_name*, *restore_suite* and *model_suffix*: are ignored if not specified

### Notes
* **The code is provided as is, please expect minimal-to-none support on it.**
* This code is provided for research purposes only. Although we tried our 
  best to test it, the code might be bugged or unstable. Use it at your own
  risk!
* PRs welcome!! Feel free to contribute to the code with a PR if you find bugs 
  or want to improve the existing code!

 
</br>
</br>
</br>

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
