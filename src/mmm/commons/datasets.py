from typing import Any, Dict

from kedro.io.core import AbstractDataset, DatasetError
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM


class DelayedSaturatedMMMModel(AbstractDataset[Any, Any]):
    def __init__(
        self,
        filepath: str,
    ) -> None:
        self._filepath = filepath

    def _describe(self) -> Dict[str, Any]:
        return {
            "filepath": self._filepath,
        }

    def _load(self) -> Any:
        return DelayedSaturatedMMM.load(self._filepath)

    def _save(self, model: Any) -> None:
        model.save(self._filepath)

    def _exists(self) -> bool:
        try:
            DelayedSaturatedMMM.load(self._filepath)
            return True
        except DatasetError:
            return False
