import os
import time

import tensorflow as tf
from tensorflow.python.training.training import SessionRunHook


class EarlyStopHook(SessionRunHook):
    def __init__(self, experiment):
        self.__name__ = 'EarlyStopHook'
        self.exp = experiment
        self.cfg = self.exp.cfg
        self.patience = self.cfg.patience
        self.val_skip = max(1, self.cfg.val_every_epochs) - 1
        self.best_score = 0
        self.metrics_history = {}
        self.validate_fn = getattr(experiment, "validate_fn", None)
        self._disable = False

    def after_run(self, run_context, run_values):
        last_epoch = False
        estop = False

        # We are not calling session.run to run the model. We can skip this.
        if not hasattr(self.exp, 'epoch_id'):
            return

        # We are calling session.run in validation. Skip.
        if self._disable:
            return

        # We hit the max number of epochs.
        if self.exp.epoch_id == self.cfg.max_epochs - 1:
            tf.logging.info('REACHED LAST EPOCH!!!')
            last_epoch = True

        # Patience is over, validate one last time and possibly early stop.
        if self.exp.epoch_id >= self.cfg.min_epochs and self.patience == 0:
            estop = True

        # Validate if last epoch, early stop or valid_every iterations passed
        if callable(self.validate_fn) and (
             last_epoch or estop or not self.val_skip):

            # Run validate on each validation set
            metrics_val = {}
            self._disable = True
            for s in self.cfg.val_on_sets:
                metrics_val[s] = self.validate_fn(
                    self.exp.val_graph_outs[s],
                    which_set=s)
                self.metrics_history.setdefault(s, []).append(metrics_val[s])
            self.exp.metrics_val = metrics_val

            valid_score = metrics_val.get('valid')
            # We improved the *validation* metric
            if valid_score >= self.best_score:
                self.best_score = valid_score
                tf.logging.info('## New best model found! Score: {} ##'.format(
                    valid_score))
                t_save = time()
                # Save best model as a separate checkpoint
                self.exp.saver.save(self.exp.sess,
                                    os.path.join(self.cfg.save_path,
                                                 'best.ckpt'),
                                    global_step=self.exp.global_step)
                t_save = time() - t_save
                tf.logging.info('Best checkpoint saved in {}s'.format(t_save))

                self.patience = self.cfg.patience  # Reset patience
                self.estop = False  # Stop potential early stopping
            else:
                self.patience -= 1
                if estop:
                    last_epoch = True
                    tf.logging.info('EARLY STOPPING!!!')
            # Start skipping again
            self.val_skip = max(1, self.cfg.val_every_epochs) - 1
            self._disable = False

        if last_epoch:
            best = self.best_score
            self.exp.return_value = self.best_score
            tf.logging.info('\nBest validation score: {:.5f}\n'.format(best))
            run_context.request_stop()  # Exit epoch loop
