import os
from time import time

import tensorflow as tf
from tensorflow.python.training.training import SessionRunHook


class EarlyStopHook(SessionRunHook):
    def __init__(self, experiment):
        self.__name__ = 'EarlyStopHook'
        self.exp = experiment
        self.cfg = self.exp.cfg
        self.patience = self.cfg.patience
        self.best_score = 0
        self.metrics_history = {}
        self.validate_fn = getattr(experiment, "validate_fn", None)
        self.saver = tf.train.Saver(
            name='BestSaver',
            save_relative_paths=True,
            max_to_keep=self.cfg.checkpoints_to_keep)

    def after_run(self, run_context, run_values):
        if not hasattr(self.exp, 'global_step_val'):
            return

        cfg = self.exp.cfg
        exp = self.exp
        nbatches = exp.train.nbatches

        # Skip validation if it's not the end of an epoch
        if (exp.global_step_val + 1) % nbatches:
            return

        # Skip validation if we did not run at least `val_skip_first` epochs
        if exp.global_step_val < (cfg.val_skip_first * nbatches):
            tf.logging.info('Skipping validation for the first %d epochs' %
                            cfg.val_skip_first)
            return

        # Skip validation if the epoch is not a multiple of `val_every_epochs`
        if (exp.global_step_val + 1) % (cfg.val_every_epochs * nbatches):
            tf.logging.info('Skipping validation: validating every %d epochs' %
                            cfg.val_every_epochs)
            return

        last_epoch = False
        estop = False

        # We hit the max number of epochs.
        if exp.epoch_id == cfg.max_epochs - 1:
            tf.logging.info('STOP TRAINING: max epoch reached!!!')
            last_epoch = True

        # Patience is over, validate one last time and possibly early stop.
        if exp.epoch_id >= cfg.min_epochs and self.patience == 0:
            estop = True

        if callable(self.validate_fn):
            # Run validate on each validation set
            metrics_val = {}
            for s in cfg.val_on_sets:
                metrics_val[s] = self.validate_fn(
                    exp.val_graph_outs[s],
                    which_set=s)
                self.metrics_history.setdefault(s, []).append(metrics_val[s])
            exp.metrics_val = metrics_val

            valid_score = metrics_val.get('valid')
            # We improved the *validation* metric
            if valid_score >= self.best_score:
                self.best_score = valid_score
                tf.logging.info('## New best model found! Score: {} ##'.format(
                    valid_score))
                t_save = time()
                # Save best model as a separate checkpoint
                self.saver.save(run_context.session,
                                os.path.join(cfg.save_path, 'best.ckpt'),
                                global_step=exp.global_step)
                t_save = time() - t_save
                tf.logging.info('Best checkpoint saved in {}s'.format(t_save))

                self.patience = cfg.patience  # Reset patience
                self.estop = False  # Stop potential early stopping
            else:
                self.patience -= 1
                if estop:
                    last_epoch = True
                    tf.logging.info('STOP TRAINING: early stopping!!!')

        if last_epoch:
            best = self.best_score
            exp.return_value = self.best_score
            tf.logging.info('\nBest validation score: {:.5f}\n'.format(best))
            run_context.request_stop()  # Exit epoch loop
