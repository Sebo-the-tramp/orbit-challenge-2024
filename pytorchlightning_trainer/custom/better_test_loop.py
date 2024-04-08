from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationEpochLoop
from pytorch_lightning.trainer.progress import BatchProgress
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.auto_restart import (
    _collect_states_on_rank_zero_over_collection,
    _reload_dataloader_state_dict,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import AbstractDataFetcher, DataLoaderIterDataFetcher
from pytorch_lightning.utilities.imports import _fault_tolerant_training
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

class BetterTestEpochLoop(EvaluationEpochLoop):
    def __init__(self) -> None:
        super().__init__()
        self.batch_progress = BatchProgress()

        self._outputs: EPOCH_OUTPUT = []
        self._dl_max_batches = 0
        self._data_fetcher: Optional[AbstractDataFetcher] = None
        self._dataloader_state_dict: Dict[str, Any] = {}
        self._dl_batch_idx = [0]

        print("INIT CI SIAMOO ", "##"*10)

    def advance(
            self,
            data_fetcher: AbstractDataFetcher,
            dl_max_batches: int,
            kwargs: OrderedDict,
        ) -> None:
            """Calls the evaluation step with the corresponding hooks and updates the logger connector.

            Args:
                data_fetcher: iterator over the dataloader
                dl_max_batches: maximum number of batches the dataloader can produce
                kwargs: the kwargs passed down to the hooks.

            Raises:
                StopIteration: If the current batch is None
            """            


            if not isinstance(data_fetcher, DataLoaderIterDataFetcher):
                batch_idx = self.batch_progress.current.ready
                batch = next(data_fetcher)
            else:
                batch_idx, batch = next(data_fetcher)
            self.batch_progress.is_last_batch = data_fetcher.done

            # check if the data is on the GPU or not

            print("WHERE IS DATATTATTATA?", next(iter(batch.values())).device)

            # configure step_kwargs
            kwargs = self._build_kwargs(kwargs, batch, batch_idx)

            self.batch_progress.increment_ready()

            # hook
            self._on_evaluation_batch_start(**kwargs)

            self.batch_progress.increment_started()

            # print(torch.cuda.memory_summary())


            # lightning module methods
            output = self._evaluation_step(**kwargs)
            output = self._evaluation_step_end(output)

            self.batch_progress.increment_processed()

            # track loss history
            self._on_evaluation_batch_end(output, **kwargs)

            self.batch_progress.increment_completed()

            # log batch metrics
            if not self.trainer.sanity_checking:
                dataloader_idx = kwargs.get("dataloader_idx", 0)
                self.trainer._logger_connector.update_eval_step_metrics(self._dl_batch_idx[dataloader_idx])
                self._dl_batch_idx[dataloader_idx] += 1

            # track epoch level outputs
            if self._should_track_batch_outputs_for_epoch_end() and output is not None:
                self._outputs.append(output)

            if self.trainer.move_metrics_to_cpu:
                # the evaluation step output is not moved as they are not considered "metrics"
                assert self.trainer._results is not None
                self.trainer._results.cpu()

            if not self.batch_progress.is_last_batch:
                # if fault tolerant is enabled and process has been notified, exit.
                self.trainer._exit_gracefully_on_signal()
