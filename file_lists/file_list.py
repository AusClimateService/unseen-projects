"""Create variable file lists for DCPP models (excluding CAFE).

Requires intake package and group dk92 membership.

"""

from dataclasses import dataclass, field
from glob import glob
from intake import open_esm_datastore
import logging
import numpy as np
import os
import re
from pathlib import Path

home = Path("/g/data/xv83/unseen-projects/code/file_lists")

# Selected DCPP Models
dcpp_models = np.array(
    [
        "BCC-CSM2-MR",
        "CanESM5",
        "CMCC-CM2-SR5",
        "EC-Earth3",
        "HadGEM3-GC31-MM",
        "IPSL-CM6A-LR",
        "MIROC6",
        "MPI-ESM1-2-HR",
        "MRI-ESM2-0",
        "NorCPM1",
    ]
)


def get_logger(name="file_list_log", level=logging.INFO):
    """Initialise a logger that writes to a stream and file."""
    logging.basicConfig(level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add file handler
    if not logger.handlers:
        handler = logging.FileHandler(f"{home}/{name}.txt")
        handler.setLevel(level)
        fformat = logging.Formatter("{asctime} {message}", "%Y-%m-%d %H:%M", style="{")
        handler.setFormatter(fformat)
        logger.addHandler(handler)
    return logger


def natsorted(a):
    """Sort a list of strings in natural order."""
    if not isinstance(a, list):
        a = list(a)

    def convert(text):
        """Convert numerical strings in list to floats."""
        return float(text) if text.isdigit() else text

    def alphanum_key(key):
        """Splits the string into a list of numbers and text."""
        return [convert(c) for c in re.split("([-+]?[0-9]*\.?[0-9]*)", key)]

    a.sort(key=alphanum_key)
    return a


def create_file_lists(cmip6, var, exclude_years={}, exclude_members={}, **kwargs):
    """Create variable file lists for each available DCPP model.

    Parameters
    ----------
    cmip6 : intake.esm.datastore
        CMIP6 data store.
    var : str
        Variable ID.
    exclude_members : dict, optional
        Model ensemble members to exclude (e.g., {"CanESM5": ["r21i1p2f1", "r22"]}).
    **kwargs : dict
        Additional search criteria.
    """
    logger = get_logger()
    subset = cmip6.search(experiment_id="dcppA-hindcast", variable_id=var, **kwargs)

    # Available models that are in dcpp_models
    models = np.unique(subset.df["source_id"])
    models = np.array([m for m in models if m in dcpp_models])

    missing_models = set(dcpp_models) - set(models)
    if len(missing_models) != 0:
        logger.info(f"{var} missing models: {missing_models}")

    for model in models:
        file = home / f"{model}_dcppA-hindcast_{var}_files.txt"
        try:
            os.remove(file)
        except OSError:
            pass

        # Subset model
        df = subset.df[subset.df["source_id"] == model]
        df = df.sort_values("member_id")

        # Determine if multiple grid types are available
        grid_label = np.unique(df["grid_label"])
        if len(grid_label) > 1:
            # Select native grid
            df = df[df["grid_label"] == "gn"]
            grid_label = "gn"
        else:
            grid_label = grid_label[0]

        # Determine if multiple realms are available
        realm = np.unique(df["realm"])
        assert len(realm) == 1, f"{model} {var}: select realm: {realm}"
        realm = realm[0]

        # Determine if multiple frequencies are available
        frequency = np.unique(df["frequency"])
        assert len(frequency) == 1, f"{model} {var}: select frequency: {frequency}"
        frequency = frequency[0]

        # Split ensemble member IDs (sYYYY-r??i??p??f??)
        df["year"] = np.array([m[1:5] for m in df["member_id"]])
        df["member"] = np.array([m[6:] for m in df["member_id"]])
        avail_years = f"{df.year.min()}-{df.year.max()}"
        avail_members = np.unique(df["member"]).size

        # Exclude any years
        if model in exclude_years:
            if type(exclude_years[model]) is not list:
                exclude_years[model] = [exclude_years[model]]
            # Convert years to strings to match df["year"]
            exclude_years[model] = [str(y) for y in exclude_years[model]]
            df = df[~df["year"].isin(exclude_years[model])]

        # Exclude any ensemble members
        if model in exclude_members:
            if type(exclude_members[model]) is not list:
                exclude_members[model] = [exclude_members[model]]
            for m in exclude_members[model]:
                df = df[~df["member"].str.contains(m)]

        # Members available each year
        years = np.unique(df.year)
        # Ensemble members available for each year
        m_year = np.array(
            [np.unique([m[6:] for m in df.member_id if m[1:5] in y]) for y in years]
        )
        m_len_year = np.array([len(m) for m in m_year])

        # # Select years with same number of ensemble members as first year
        # # Note: better to use exclude_years as this will drop years with any missing members
        # years = years[m_len_year >= years_count[0]]
        # m_year = m_year[years_count >= years_count[0]]

        # Check for missing years
        assert np.diff([int(y) for y in years]).all() == 1

        # Select ensemble members available for all selected years
        all_members = natsorted(np.unique(df.member))
        members = [m for m in all_members if all([m in y for y in m_year])]

        members = natsorted(members)
        ensemble_size = len(members)
        assert (
            ensemble_size > 0
        ), f"{model} {var} {years[0]}-{years[-1]} ensembles/year={m_len_year}"

        # Log number of files per year for each ensemble member
        n_files_year = []

        # Iterate years
        for t in years:
            n_files = []

            df_t = df[df.year == t]
            member_ids_t = np.unique([m for m in df_t.member_id if m[6:] in members])
            member_ids_t = natsorted(member_ids_t)
            assert (
                natsorted(np.unique([m[6:] for m in member_ids_t])) == members
            ), f"Error: {model} {var} year {t} missing members"

            # Iterate ensemble member IDs (sYYYY-r??i??p??f??)
            for m in member_ids_t:
                df_m = df_t[df_t["member_id"] == m]

                # Select path of latest version
                version = np.unique(df_m["version"])[-1]
                df_m = df_m[df_m["version"] == version]
                assert np.unique(df_m["version"]).size == 1

                paths = natsorted(df_m["path"].values)
                n_files.append(len(paths))

                # Write to file
                with open(file, "a") as outfile:
                    for item in paths:
                        outfile.write(f"{item}\n")
            n_files_year.append(sum(n_files))

        n_time_files = np.unique(n_files_year)

        # Check consistent number of files for each year and ensemble member
        assert n_time_files.size == 1, f"{model} n_time_files={n_time_files}"
        # Check number of files per year is consistent with ensemble size
        assert n_time_files[0] == ensemble_size * np.unique(n_files)[0]
        # Log model ensemble size and number of files per year
        logger.info(
            f"{model:<15s} {var:<6s} realm={realm} freq={frequency} grid={grid_label} members={ensemble_size:>2d}/{avail_members:<2d} years={years[0]}-{years[-1]} (all={avail_years}) files/year={n_time_files[0]:>3d}"
        )
    logger.info("")


if __name__ == "__main__":

    cmip6 = open_esm_datastore("/g/data/dk92/catalog/v2/esm/cmip6-oi10/catalog.json")

    # CanESM5: exclude ensemble members r21*-r40* (to match pr available members)
    # EC-Earth3: From 2005-2010 there are no pr r1i4p1f1 files, so i4 has been left out
    exclude_members = {
        "pr": {},
        "tos": {"CanESM5": [f"r{i}" for i in range(21, 41)], "EC-Earth3": ["i4"]},
        "tasmax": {},
    }
    # CanESM5: exclude years 2017-2019 (to match pr available 1960-2016)
    # EC-Earth3: pr 2018 is left out because it does not have i2 files
    exclude_years = {
        "pr": {"EC-Earth3": ["2018", "2019"]},
        "tos": {
            "CanESM5": ["2017", "2018", "2019"],
            "EC-Earth3": ["2018"],
            "MIROC6": ["2019", "2020", "2021"],
        },
        "tasmax": {"EC-Earth3": ["2018", "2019"]},
    }
    create_file_lists(
        cmip6,
        var="pr",
        frequency="day",
        exclude_years=exclude_years["pr"],
        exclude_members=exclude_members["pr"],
    )

    create_file_lists(
        cmip6,
        var="tos",
        exclude_years=exclude_years["tos"],
        exclude_members=exclude_members["tos"],
        frequency="mon",
        realm="ocean",
    )
    create_file_lists(
        cmip6,
        var="tasmax",
        frequency="day",
        exclude_years=exclude_years["tasmax"],
        exclude_members=exclude_members["tasmax"],
    )
