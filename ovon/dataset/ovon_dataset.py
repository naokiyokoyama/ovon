import os
from typing import List, Optional

import attr
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode


@attr.s(auto_attribs=True, kw_only=True)
class OVONEpisode(ObjectGoalNavEpisode):
    r"""OVON Episode

    :param children_object_categories: Category of the object
    """
    children_object_categories: Optional[List[str]] = []

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"
