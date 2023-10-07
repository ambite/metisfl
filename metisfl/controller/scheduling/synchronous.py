
""" MetisFL Synchronous Scheduler """


from typing import List

from loguru import logger

from metisfl.controller.scheduling.scheduler import Scheduler


class SynchronousScheduler(Scheduler):

    learner_ids: set = set()
    global_iteration: int = 0

    def schedule(
        self,
        learner_id: str,
    ) -> List[str]:
        """Schedule the next batch of learners, synchronously.

        Parameters
        ----------
        learner_id : str
            The ID of the learner to schedule.

        Returns
        -------
        List[str]
            The IDs of the learners to schedule.
        """

        self.learner_ids.add(learner_id)

        if len(self.learner_ids) < self.num_learners:
            return []

        self.global_iteration += 1

        logger.info(f"Federation round {self.global_iteration}.")

        to_schedule = self.learner_ids.copy()
        
        self.learner_ids.clear()

        return list(to_schedule)

    def __str__(self) -> str:
        return "SynchronousScheduler"
