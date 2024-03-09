"""Main module for the application."""

from emulode.emulation import EmulationFactory

if __name__ == "__main__":
    emulation = EmulationFactory.create_from_yml_file("config.yml", ideal_run=False)
    emulation.plot()
