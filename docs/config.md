# Configuration classes

The configuration parameters provided in the configuration files are used to create the `Configs` class object. This object is used to store the configuration parameters and to provide the configuration parameters to the other classes.

## Configs class

::: emulode.config.Configs
    options:
        members:
        - 

---

The `ode`, `solver`, `simulation`, `emulation` and `plotter` members are instances of the `Config` abstract base class.

## Config abstract base class

::: emulode.config.Config
    options:
        members:
        - 

The `Config` abstract base class is used to define the configuration parameters for the `ode`, `solver`, `simulation`, `emulation` and `plotter` members of the `Configs` class.

> **Note:** In the following classes, all parameters other than `config_dict` are optional. The `config_dict` parameter is a dictionary containing the default configuration parameters (as typically read from a configuration file). The optional parameters if provided will override the parameters in the `config_dict` dictionary.

---

## ODEConfig class

::: emulode.config.ODEConfig
    options:
        members:
        -

---

## SolverConfig class

::: emulode.config.SolverConfig
    options:
        members:
        -

---

## SimulatorConfig class

::: emulode.config.SimulatorConfig
    options:
        members:
        -

---

## EmulatorConfig class

::: emulode.config.EmulatorConfig
    options:
        members:
        -

---

## PlotterConfig class

::: emulode.config.PlotterConfig
    options:
        members:
        -

