"""Unchanged registered J0 convergence policy."""
import copy
import math
def _finite(value,key):
    value=float(value)
    if not math.isfinite(value): raise ValueError("Nonfinite "+key)
    return value

class PlateauController:
    """Any substantive component improvement resets a deterministic plateau.

    Tiny changes accumulate against the last substantive reference. Learning
    rate reductions are milestones of each uninterrupted plateau, not each
    epoch. A successful convergence requires actual post-warmup validation.
    """
    def __init__(self, phase, min_epochs=2, patience=8, absolute_delta=.002,
                 relative_delta=.005, reductions=(3, 6), min_optimizer_steps=0,
                 post_lr_drop_epochs=0):
        if phase not in ('A', 'B', 'C') or min_epochs < 0 or patience < max(reductions):
            raise ValueError('Invalid plateau configuration')
        self.phase = phase
        self.min_epochs = float(min_epochs)
        self.patience = int(patience)
        self.absolute_delta = float(absolute_delta)
        self.relative_delta = float(relative_delta)
        self.reductions = tuple(int(x) for x in reductions)
        self.best = {}
        self.bad_checks = 0
        self.total_reductions = 0
        self.checks = 0
        self.converged = False
        self.min_optimizer_steps = int(min_optimizer_steps)
        self.post_lr_drop_epochs = float(post_lr_drop_epochs)
        self.last_lr_drop_epoch = None
        self.last_optimizer_steps = 0
        if self.min_optimizer_steps < 0 or self.post_lr_drop_epochs < 0:
            raise ValueError('Negative minimum optimization budget')

    def observe(self, epoch, values, optimizer_steps=0):
        values = {k: _finite(v, k) for k, v in values.items()}
        if not values or (self.best and set(values) != set(self.best)):
            raise ValueError('Plateau metric axis changed')
        self.checks += 1
        if int(optimizer_steps) < self.last_optimizer_steps:
            raise ValueError('Optimizer step count went backwards')
        self.last_optimizer_steps = int(optimizer_steps)
        improved = not self.best
        for key, value in values.items():
            threshold = self.absolute_delta if self.phase == 'A' else self.relative_delta * max(abs(self.best.get(key, value)), 1e-6)
            if key not in self.best or self.best[key] - value >= threshold:
                self.best[key] = value
                improved = True
        if improved:
            self.bad_checks = 0
        elif float(epoch) >= self.min_epochs:
            self.bad_checks += 1
        reduce_lr = self.bad_checks in self.reductions and not improved and float(epoch) >= self.min_epochs
        if reduce_lr:
            self.total_reductions += 1
            self.last_lr_drop_epoch = float(epoch)
        post_drop_complete = (self.last_lr_drop_epoch is not None and
                              float(epoch) - self.last_lr_drop_epoch >= self.post_lr_drop_epochs)
        self.converged = (float(epoch) >= self.min_epochs and self.bad_checks >= self.patience
                          and self.total_reductions >= len(self.reductions)
                          and self.last_optimizer_steps >= self.min_optimizer_steps and post_drop_complete)
        return dict(improved=improved, reduce_lr=reduce_lr, converged=self.converged,
                    bad_checks=self.bad_checks, total_reductions=self.total_reductions,
                    minimum_updates_met=self.last_optimizer_steps >= self.min_optimizer_steps,
                    post_lr_drop_epochs_met=post_drop_complete)

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_state(cls, state):
        out = cls(state['phase'], state['min_epochs'], state['patience'],
                  state['absolute_delta'], state['relative_delta'], state['reductions'],
                  state.get('min_optimizer_steps', 0), state.get('post_lr_drop_epochs', 0))
        out.__dict__.update(copy.deepcopy(state))
        out.reductions = tuple(out.reductions)
        return out

class JointPlateauController(PlateauController):
    """A fixed scalar output objective; no candidate-dependent ranks."""
    def __init__(self, min_epochs=10, patience=6, relative_delta=.002,
                 reductions=(2,4), min_optimizer_steps=1500, post_lr_drop_epochs=2):
        super().__init__('B', min_epochs=min_epochs, patience=patience,
            relative_delta=relative_delta, reductions=reductions,
            min_optimizer_steps=min_optimizer_steps, post_lr_drop_epochs=post_lr_drop_epochs)
        self.joint_objective = True

    @classmethod
    def from_state(cls, state):
        out = cls(state['min_epochs'], state['patience'], state['relative_delta'],
            state['reductions'], state['min_optimizer_steps'], state['post_lr_drop_epochs'])
        out.__dict__.update(copy.deepcopy(state)); out.reductions = tuple(out.reductions)
        return out
