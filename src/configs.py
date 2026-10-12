import os
import pathlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import yaml  # type: ignore

# import torch # type: ignore


### Base Configurator
class Configurator:
    FIRST_LOAD = False

    def __init__(
        self, config_fp: str, config_type: str = "yaml", with_partition=False
    ) -> None:
        """Configuration file loader.

        Args:
            config_type: Type of configuration file, either 'yaml' or 'torch'. Defaults to 'yaml'.
            with_partition: If configuration file is a partitioned yaml file. Defaults to False.
        """
        if Configurator.FIRST_LOAD:
            print(
                f'{self.__class__.__name__} Loading configuration from "{config_fp}".'
            )
            Configurator.FIRST_LOAD = False

        if config_type == "yaml":
            if with_partition:
                yaml_load = self.from_yaml_all(config_fp)
            else:
                yaml_load = self.from_yaml(config_fp)
            for key in yaml_load:
                setattr(self, key, yaml_load[key])


    def set_extra_attr(self, key, value):
        setattr(self, key, value)

    @staticmethod
    def from_yaml(load_path) -> Union[dict, Any]:
        with open(load_path, "r") as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml

    @staticmethod
    def from_yaml_all(load_path) -> Union[dict, Any]:
        parsed_yaml = {}
        with open(load_path, "r") as stream:
            try:
                for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                    parsed_yaml.update(data)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml



class _Configuration(ABC):
    """Base class for configuration/specification classes."""

    def __init__(self, config: Configurator, manual_load=False) -> None:
        self._config = config
        if not manual_load:
            self._load_config()

    @abstractmethod
    def _load_config(self):
        pass

    def manual_load(self):
        self._load_config()

    @classmethod
    def from_yaml(cls, yaml_fp: str, with_partition=False):
        config = Configurator(yaml_fp, with_partition=with_partition)
        return cls(config)



### Agent Specifications
class CircularRobotSpecification(_Configuration):
    """Specification class for circular robots."""

    def __init__(self, config: Configurator):
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts  # sampling time

        self.vehicle_width = config.vehicle_width
        self.vehicle_margin = config.vehicle_margin
        self.social_margin = config.social_margin
        self.lin_vel_min = config.lin_vel_min
        self.lin_vel_max = config.lin_vel_max
        self.lin_acc_min = config.lin_acc_min
        self.lin_acc_max = config.lin_acc_max
        self.ang_vel_max = config.ang_vel_max
        self.ang_acc_max = config.ang_acc_max
        self.ang_vel_onm_max = config.ang_vel_onm_max
        self.lidar_range = config.lidar_range


class PedestrianSpecification(_Configuration):
    """Specification class for pedestrians."""

    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts  # sampling time

        self.human_width = config.human_width
        self.human_vel_max = config.human_vel_max
        self.human_stagger = config.human_stagger



class GPDFConfiguration(_Configuration):
    """Configuration class for GPDF Module."""

    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.d_offset_0 = (
            config.D_OFFSET_0
        )  # offset distance from the original boundary
        self.L = config.L  # interpolation value to determine the closeness of points
        self.rho = config.Rho
        self.rho_uni = config.Rho_Uni


class EnvContrConfiguration(_Configuration):
    """Configuration class for GPDF Module."""

    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.dynamics = config.dynamics
        self.env_name = config.env_name
        self.init_state = config.init_state
        self.target = np.array(config.target)
        self.num_dyn_circle = config.num_dyn_circle
        self.radius = config.radius
        self.threeD_controller = config.threeD_controller
        self.MCBF = config.MCBF
        self.autotune = config.autotune
        self.nominal_speed = config.nominal_speed

        if self.autotune:
            if self.threeD_controller:
                if self.env_name == "hospital":
                    contr_path = os.path.join(
                        pathlib.Path(__file__).resolve().parents[1],
                        "config",
                        "autotune_3D_" + self.env_name + ".yaml",
                    )
            else:
                contr_path = os.path.join(
                    pathlib.Path(__file__).resolve().parents[1],
                    "config",
                    "autotune_2D.yaml",
                )
        else:
            if (self.target == [-5.0, 0.0]).all() and self.env_name == "hospital":
                scenario = 1
            elif (self.target == [-10.0, 0.0]).all() and self.env_name == "hospital":
                scenario = 2
            elif (self.target == [-14.0, 0.0]).all() and self.env_name == "hospital":
                scenario = 3
            elif (self.target == [-10.0, -5.5]).all() and self.env_name == "hospital":
                scenario = 4
            elif (self.target == [-6.0, 5.5]).all() and self.env_name == "hospital":
                scenario = 5
            else:
                scenario = 1
                warnings.warn(
                    "This target is not supported in handtune mode.", RuntimeWarning
                )

            if self.threeD_controller:
                contr_path = os.path.join(
                    pathlib.Path(__file__).resolve().parents[1],
                    "config/handtune/3D_controller",
                    self.env_name + "_3Dcontroller_scenario" + str(scenario) + ".yaml",
                )
            else:
                contr_path = os.path.join(
                    pathlib.Path(__file__).resolve().parents[1],
                    "config/handtune/2D_controller",
                    self.env_name + "_2Dcontroller_scenario" + str(scenario) + ".yaml",
                )

        contr = ContrConfiguration.from_yaml(contr_path)
        self.w = contr.w
        self.iter = contr.iter
        self.iter_g = contr.iter_g
        self.iter_c = contr.iter_c
        self.hold_time = contr.hold_time
        self.base_margin = contr.base_margin
        self.target_range = contr.target_range
        self.om_range = contr.om_range
        self.ve = contr.ve
        self.beta_coef = contr.beta_coef
        self.om_expand_threshold = contr.om_expand_threshold
        self.dir_num = contr.dir_num
        self.custom_iter1 = contr.custom_iter1
        self.custom_iter2 = contr.custom_iter2


class ContrConfiguration(_Configuration):
    """Configuration class for GPDF Module."""

    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.w = config.w
        self.iter = config.iter
        self.iter_g = config.iter_g
        self.iter_c = config.iter_c
        self.hold_time = config.hold_time
        self.base_margin = config.base_margin
        self.target_range = config.target_range
        self.om_range = config.om_range
        self.ve = config.ve
        self.beta_coef = config.beta_coef
        self.om_expand_threshold = config.om_expand_threshold
        self.dir_num = config.dir_num
        self.custom_iter1 = config.custom_iter1
        self.custom_iter2 = config.custom_iter2


