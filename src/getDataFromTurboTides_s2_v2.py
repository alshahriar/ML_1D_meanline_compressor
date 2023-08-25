# -*- coding: utf-8 -*-
"""
-------------------------------------------------
  File Name：   getDataFromTurboTides
  Description :
  Author :    GYH
  date：     2023-07-11 08:11
-------------------------------------------------
  Change Activity:
          2023-07-11 08:11
-------------------------------------------------
"""
__author__ = 'GYH'

import os
import sys
import time
import subprocess as sp
import re
from pathlib import Path
import json
from TurboTides import tt


class GetDataFromTurboTides:
    def __init__(self):
        self.TT_API = r'C:\TurboTides\gtts_api.bat'
        self.inputJsonParametersFile = r'inputParameters.json'
        self.outputJsonParametersFile = r'OutputParameters.json'
        self.outPutRetJsonFilePath = r'outPutRet.json'
        self.file = r'test_api.tml'
    
    def startTurboTidesWithApi(self):
        # envPath=r'C:\TurboTides\env.bat'
        envPath = Path(self.TT_API).parent / "env.bat"
        # exePath=r'C:\TurboTides\dlls\gtts\TurboTides.exe'
        exePath = Path(self.TT_API).parent / "dlls" / "gtts" / "TurboTides.exe"
        tt.launch(str(envPath), str(exePath))
    
    def setParameters(self):
        tt.cdm("1D")
        # obj = tt.get_object("1d")
        with open(self.inputJsonParametersFile, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            cmd = f'o=cds();o.property("{key}","{value}")'
            tt.run_js(cmd)
            print(cmd)
        solve_cmd = r'cds();o=cd("1d");o.solve()'
        tt.run_js(solve_cmd)
        return 0
    
    def getParameters(self):
        tt.cdm("1D")
        with open(self.outputJsonParametersFile, "r") as f:
            data = json.load(f)
        exportData = {}
        for key, value in data.items():
            cmd = f'o=cds();o.property("{value}")'
            ret = tt.run_js(cmd).get('text')
            exportData[key] = ret
        with open(self.outPutRetJsonFilePath, 'w') as f:
            json.dump(exportData, f)
        return 0
    
    def main(self):
        self.startTurboTidesWithApi()
        tt.load(str(Path(self.file).absolute()), True)
        self.setParameters()
        self.getParameters()
        # tt.exit()
        return 0


def main():
    getData = GetDataFromTurboTides().main()


if __name__ == '__main__':
    main()
