import argparse

from .config import Settings


class Cli:
    def __init__(self):
        parser = argparse.ArgumentParser(prog="neodraw")
        parser.add_argument("command", nargs="?", default="run", choices=("run", "check", "fetch"))
        parser.add_argument("--camera")
        self.args = parser.parse_args()
        self.settings = Settings.load(camera=self.args.camera)

    def run(self):
        from .app import NeoDrawApp
        NeoDrawApp(self.settings).run()

    def check(self):
        from .check import HardwareCheck
        HardwareCheck(self.settings).run()

    def fetch(self):
        from .assets import Assets
        Assets(self.settings).fetch()

    def main(self):
        getattr(self, self.args.command)()


if __name__ == "__main__":
    Cli().main()
