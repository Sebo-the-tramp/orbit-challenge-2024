import os
import shutil
import sys
from collections import ChainMap, OrderedDict
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

import torch

from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.loops.dataloader import DataLoaderLoop
from pytorch_lightning.loops.epoch import EvaluationEpochLoop
from pytorch_lightning.loops.utilities import _set_sampler_epoch
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, _ResultCollection
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import (
    AbstractDataFetcher,
    DataFetcher,
    DataLoaderIterDataFetcher,
    InterBatchParallelDataFetcher,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop

if _RICH_AVAILABLE:
    from rich import get_console
    from rich.table import Column, Table


class BetterLoop(EvaluationLoop):
    """Loops over all dataloaders for evaluation."""

    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.epoch_loop = EvaluationEpochLoop()
        self.verbose = verbose

        self._results = _ResultCollection(training=False)
        self._outputs: List[EPOCH_OUTPUT] = []
        self._logged_outputs: List[_OUT_DICT] = []
        self._max_batches: List[Union[int, float]] = []
        self._has_run: bool = False
        self._data_fetcher: Optional[AbstractDataFetcher] = None


    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Performs evaluation on one single dataloader."""

        print("##"*10, "NEW EVALUATION LOOOOOOP")

        dataloader_idx = self.current_dataloader_idx
        dataloader = self.current_dataloader

        # it = next(iter(dataloader))

        # print(it["category_names"])

        # for object_item in dataloader.

        def batch_to_device(batch: Any) -> Any:
            batch = self.trainer.lightning_module._on_before_batch_transfer(batch, dataloader_idx=dataloader_idx)            
            batch = self.trainer._call_strategy_hook("batch_to_device", batch, dataloader_idx=dataloader_idx)                        
            return batch

        # for object_item in it["category_names"]:
                   
        assert self._data_fetcher is not None
        self._data_fetcher.setup(dataloader, batch_to_device=batch_to_device)
        dl_max_batches = self._max_batches[dataloader_idx]

        kwargs = OrderedDict()
        if self.num_dataloaders > 1:
            kwargs["dataloader_idx"] = dataloader_idx

        # print(torch.cuda.memory_summary())

        dl_outputs = self.epoch_loop.run(self._data_fetcher, dl_max_batches, kwargs)

        # print(torch.cuda.memory_summary())

        # store batch level output per dataloader
        self._outputs.append(dl_outputs)

        if not self.trainer.sanity_checking:
            # indicate the loop has run
            self._has_run = True