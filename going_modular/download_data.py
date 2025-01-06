"""
Contains functionality to dowload zipped data from github repo and store it
locally.
"""
import os
import pathlib
import requests
import zipfile
from pathlib import Path

def download_data(project_dir : str,
                  root_dir: str,
                  github_url: str) -> None:
  """Dowloads zipped data locally from the specified github url and extracts it.

  Downloads a zipped data, extracts it locally and also removes the zipped file
  and other uncessary files to clean up.

  Args:
    project_dir: The name of the poject dir that will host the data and scripts
        and all other all stuff.
    root_dir: The root directory where data will be stored in standard
        classification format.
    github_url: The raw github repo link in str where the zipped data will be
        downloaded.
    example usage:
           download_data(project_dir = "<project_folder_name>",
                         root_dir = "<name_of_root_dir>",
                         github_url = "<raw_url_link>"
                         )
  Returns:
      Nothing but creates a structural folder to build upon in this formart.

      --project_folder
        --data
            --pizza_steak_sushi
                  --train
                      --pizza
                        : (.png files)
                      --steak
                        : (.png files)
                      --sushi
                        : (.png files)
                  --test
                      --pizza
                        : (.png files)
                      --steak
                        : (.png files)
                      --sushi
                        : (.png files)
        --scripts
            --download_data.py
            --data_setup.py
            --model.builder.py
            --engine.py
            --train.py
            --utilities.py
            : (other .py files)
        --models
            --some_model.pth
              :(other .pth models)
        --misc
          : (other misc stuff)

  """
  # make project_dir if one doesn't exist.
  try:
    os.mkdir(project_dir)
  except FileExistsError:
    print(f"{project_dir} already exists!!! Skipping creating one")

  # check if data dir exits, make one if it doesn't.
  if Path(root_dir).is_dir():
    print(f"{root_dir} already exists! Skipping download.....")
  else:
    print(f"{root_dir} doesn't exit! Making one.......")

    # create root_dir directory
    root_dir =  Path(root_dir)
    root_dir.mkdir(parents = True, exist_ok = True)

    # join the paths -> going_modular/data/pizza_steak_sushi
    data_dir = Path(os.path.join(project_dir, root_dir, "pizza_steak_sushi"))

    # make a request to the github url for the data
    request = requests.get(github_url)
    print(f"Getting data from {github_url}...............")

    # make a zip file to store the data intermediatley
    zip_file = root_dir / "pizza_steak.sushi.zip"

    # write the requested to zip_file
    with open(zip_file, "wb") as file_writer:
      file_writer.write(request.content)
      print(f"Writting the data to {zip_file} file..........")

    # extract zipped data to data_dir
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
      print(f"Extracting the data to {data_dir}..............")
      zip_ref.extractall(data_dir)

    # remove the uncessary files and dirs for clean up
    print(f"Removing {zip_file} file and {root_dir} directory")
    os.remove(zip_file)
    os.rmdir(root_dir)

    print(f"\n[INFO] process complete!")
