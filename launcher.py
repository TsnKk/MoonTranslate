"""Capture startup failures, including imports, when launched by pythonw."""
import ctypes
import os
from pathlib import Path
import sys
import traceback


def main():
    log_path = Path(os.environ.get('TEMP', str(Path.home()))) / 'MoonTranslate-startup.log'
    try:
        log = log_path.open('w',encoding='utf-8',buffering=1)
        sys.stdout = log
        sys.stderr = log
        print('Python:',sys.executable,flush=True)
        print('Application:',Path(__file__).resolve().parent,flush=True)
        os.chdir(Path(__file__).resolve().parent)
        import app
        print('Imports OK',flush=True)
        return app.main()
    except Exception:
        error = traceback.format_exc()
        try:
            with log_path.open('a',encoding='utf-8') as output:
                output.write(error)
        except OSError:
            pass
        ctypes.windll.user32.MessageBoxW(None,
            f'MoonTranslate could not start.\n\n{error[-1800:]}\n\nLog: {log_path}',
            'MoonTranslate startup error',0x10)
        return 1


if __name__ == '__main__':
    sys.exit(main())
