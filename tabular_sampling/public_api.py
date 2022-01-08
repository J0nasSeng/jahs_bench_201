import logging
import pandas as pd
from pathlib import Path
import shutil
from typing import Optional, Union
from functools import partial

from tabular_sampling.distributed_nas_sampling import run_task
from tabular_sampling.lib.core.constants import Datasets
from tabular_sampling.lib.core.utils import DirectoryTree, MetricLogger, AttrDict

from tabular_sampling.surrogate.xgb import XGBSurrogate

_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)

def _map_dataset(dataset: str) -> Datasets:
    try:
        ds = [d for d in Datasets.__members__.values() if d.value[0] == dataset][0]
    except IndexError:
        raise ValueError(f"Invalid dataset name {dataset}, must be one of "
                         f"{[d.value[0] for d in Datasets.__members__.values()]}.")
    else:
        return ds


class Benchmark:
    def __init__(self, use_surrogate: bool = True, surrogate_kwargs: Optional[dict] = None):
        if use_surrogate:
            self.surrogate = XGBSurrogate(**surrogate_kwargs)
            self._call_fn = partial(Benchmark._benchmark_surrogate, surrogate=self.surrogate)
        else:
            self.surrogate = None
            self._call_fn = Benchmark._benchmark_live


    def __call__(self, config: dict, dataset: str = Datasets.cifar10.value[0],
                 datadir: Optional[Union[str, Path]] = None, nepochs: Optional[int] = 200,
                 batch_size: Optional[int] = 256, use_splits: Optional[bool] = True,
                 train_config: Optional[dict] = None, **kwargs):
        return self._call_fn(config=config, dataset=dataset, datadir=datadir, nepochs=nepochs, batch_size=batch_size,
                             use_splits=use_splits, train_config=train_config, **kwargs)

    @staticmethod
    def _benchmark_surrogate(surrogate: XGBSurrogate, config: dict, dataset: str, datadir: Union[str, Path],
                             nepochs: Optional[int] = 200, batch_size: Optional[int] = 256,
                             use_splits: Optional[bool] = True, train_config: Optional[dict] = None, **kwargs) -> dict:
        features = pd.DataFrame(config.values(), columns=config.keys())
        features.loc[:, "epoch"] = nepochs

        outputs = surrogate.predict(features)
        return outputs.to_dict()

    @staticmethod
    def _benchmark_live(config: dict, dataset: str, datadir: Union[str, Path], nepochs: Optional[int] = 200,
                        batch_size: Optional[int] = 256, use_splits: Optional[bool] = True,
                        train_config: Optional[dict] = None, **kwargs) -> dict:
        """ Simple wrapper around the base benchmark data generation capabilities offered by
        tabular_sampling.distributed_nas_samplig.run_task(). Providing 'train_config' and 'kwargs' dicts can be used to
        access the full range of customizations offered by 'run_task()' - consults its documentation if needed. This
        script requires access to /tmp in order to write temporary data. """

        if isinstance(datadir, str):
            datadir = Path(datadir)

        if train_config is None:
            train_config = dict(epochs=nepochs, batch_size=batch_size, use_grad_clipping=False, split=use_splits,
                                warmup_epochs=0, disable_checkpointing=True, checkpoint_interval_seconds=3600,
                                checkpoint_interval_epochs=50)

        basedir = Path("/tmp") / "tabular_sampling"
        basedir.mkdir(exist_ok=True)
        dataset = _map_dataset(dataset)

        args = {**dict(basedir=basedir, taskid=0, train_config=AttrDict(train_config), dataset=dataset, datadir=datadir,
                       local_seed=None, global_seed=None, debug=False, generate_sampling_profile=False, nsamples=1,
                       portfolio_pth=None, cycle_portfolio=False, opts=config), **kwargs}
        run_task(**args)

        dtree = DirectoryTree(basedir=basedir, taskid=0, model_idx=1, read_only=True)
        metric_pth = MetricLogger._get_sorted_metric_paths(pth=dtree.model_metrics_dir)
        df = pd.read_pickle(metric_pth)

        # Model metrics dataframe does not have a MultiIndex index - it's simply the epochs and their metrics!
        nepochs = df.index.max()
        latest = df.loc[nepochs]
        shutil.rmtree(dtree.basedir, ignore_errors=True)

        return latest.to_dict()


if __name__ == "__main__":
    from tabular_sampling.search_space.configspace import joint_config_space
    config = joint_config_space.get_default_configuration().get_dictionary()
    res = Benchmark._benchmark_live(config=config, dataset="Cifar-10", datadir="/home/archit/thesis/datasets",
                                    nepochs=3)
    print(res)
