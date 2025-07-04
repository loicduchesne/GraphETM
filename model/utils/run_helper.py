# External
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional, Union
from decorator import contextmanager

import torch

import wandb

### METRIC LOGGING
class RunManager:
    def __init__(self, wandb_run):
        self.step_metrics = defaultdict(list)
        self.epoch_metrics = {}
        self.step = 0
        self.epoch = 0

        self.wandb = wandb_run

    def next_step(self):
        self.step += 1

    def next_epoch(self):
        self.epoch += 1
        if self.step_metrics:
            self._on_epoch_end()

    def log(self, key: str, value: Union[float, torch.Tensor],
            on_step: Optional[bool] = None, on_epoch: Optional[bool] = None,
            phase: str = None, aggregate: bool = False) -> None:
        """
        Log a single metric.
        Args:
            key: Metric name.
            value: Metric value.
            on_step: Whether logging to a batch.
            on_epoch: Whether logging to an epoch.
            phase: Phase prefix (e.g.: train/val/test).
            aggregate: Whether to aggregate metrics across batches to be averaged. Default: False.
        """

        # Handle tensor conversion
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.detach().cpu().numpy().mean()

        # Create key
        full_key = f'{phase}/{key}' if phase else key

        # Determine logging behavior
        if on_step and aggregate: # TODO: Aggregate doesn't work?
            # Aggregate step-level metrics
            self.step_metrics[full_key].append(value)

        elif on_step:
            # Log step-level metrics
            self._log_to_wandb({full_key: value})

        elif on_epoch:
            # Log epoch-level metrics
            self._log_to_wandb({full_key: value})

        else:
            raise Exception(f'Either on_step or on_epoch must be True. Current: on_step={on_step}, on_epoch={on_epoch}')

    def log_dict(self, metrics: Dict[str, Any], **kwargs) -> None:
        """Log multiple metrics at once."""
        metrics_flattened = self._flatten_dict_safe(metrics, sep='/')

        for key, value in metrics_flattened.items():
            self.log(key, value, **kwargs)

    def _log_to_wandb(self, metrics: Dict[str, float]):
        """Log to wandb."""
        # Example: wandb.log(metrics)
        wandb.log(metrics, step=self.step)

    def _on_epoch_end(self):
        """Compute aggregated metrics and log to wandb."""

        # Compute epoch aggregations
        epoch_aggregated = {}
        for key, values in self.step_metrics.items():
            if values: # "values" is a list.
                epoch_aggregated[key] = np.mean(values)

        # Log aggregated metrics to wandb
        if self.wandb and epoch_aggregated:
            self.log_dict(epoch_aggregated, on_epoch=True, phase=None)

        # Reset dictionary
        self.step_metrics.clear()

    def _flatten_dict_safe(self, nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        items = []

        for key, value in nested_dict.items():
            new_key = f'{parent_key}{sep}{key}' if parent_key else key

            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                items.extend(self._flatten_dict_safe(value, new_key, sep=sep).items())
            else:
                # Handle tensor conversion immediately
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        converted_value = value.item()
                    else:
                        converted_value = value.detach().cpu().numpy().mean()
                    items.append((new_key, converted_value))
                else:
                    items.append((new_key, value))

        return dict(items)

    # WANDB CONTEXT MANAGER
    @contextmanager
    def wandb_run(self):
        """Context manager for wandb. Used to catch failed runs."""
        try:
            yield self.wandb
        except KeyboardInterrupt:
            self.wandb.finish(exit_code=2)
            raise
        except Exception:
            self.wandb.finish(exit_code=1)
            raise
        else:
            self.wandb.finish()