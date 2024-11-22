#
# SonarQube Python Plugin
# Copyright (C) 2011-2024 SonarSource SA
# mailto:info AT sonarsource DOT com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Sonar Source-Available License for more details.
#
# You should have received a copy of the Sonar Source-Available License
# along with this program; if not, see https://sonarsource.com/license/ssal/
#

import os
from utils.folder_manager import FolderManager


def test_clean_up_save_location_folder_without_removing_license(save_location_folder):
    folder_manager = FolderManager()
    folder_manager.output_folder = save_location_folder
    folder_manager.license_file = os.path.join(save_location_folder, "LICENSE")
    folder_manager.cleanup_output_folder()

    assert len(os.listdir(save_location_folder)) == 1
    assert os.listdir(save_location_folder)[0] == "LICENSE"
