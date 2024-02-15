"""Main module for the application."""

from emulode.emulation import EmulationFactory

if __name__ == "__main__":
    emulation = EmulationFactory.create("config.yml")
    emulation.plot()
