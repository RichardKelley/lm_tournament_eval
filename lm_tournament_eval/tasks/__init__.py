
from typing import Optional, Union, List


class TaskManager:


    def __init__(
        self,
        include_path : Optional[Union[str, List]] = None,
        include_defaults : bool = True
    ) -> None: 
        self.include_path = include_path
        self.include_defaults = include_defaults

        self._task_list = self.initialize_tasks(
            include_path=self.include_path, include_defaults=self.include_defaults
        )

    def initialize_tasks(self, include_path, include_defaults):
        # TODO
        pass