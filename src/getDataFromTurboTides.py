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
import pandas as pd

class GetDataFromTurboTides:
    def __init__(self):
        self.exportData = {}
        self.TT_API = r'C:\TurboTides\gtts_api.bat'
        self.inputJsonParametersFile = r'../tt_input/Batch_0.json'
        self.outputJsonParametersFile = r'OutputParameters.json'
        self.outPutRetJsonFilePath = r'../tt_output/outPutRet.json'
        self.file = r'../tt_sample/MiniTT_Baseline_Parameter_Study.tml'  # 'TurboTides/MiniTT_Baseline_Parameter_Study.tml'
        with open(self.outputJsonParametersFile, "r") as f:
            self.outdata = json.load(f)
        with open(self.inputJsonParametersFile, "r") as f:
            self.indata = json.load(f)
    
    def startTurboTidesWithApi(self):
        # envPath=r'C:\TurboTides\env.bat'
        envPath = Path(self.TT_API).parent / "env.bat"
        # exePath=r'C:\TurboTides\dlls\gtts\TurboTides.exe'
        exePath = Path(self.TT_API).parent / "dlls" / "gtts" / "TurboTides.exe"
        tt.launch(str(envPath), str(exePath))
    
    def setParameters(self):
        tt.cdm("1D")
        # obj = tt.get_object("1d")
        for key, value in self.indata.items():
            if key not in {'Object Parameter Unit', }:
                cmd = f'o=cds();o.property("{key}","{value[self.i]}")'
                tt.run_js(cmd)
        solve_cmd = r'cds();o=cd("1d");o.solve()'
        tt.run_js(solve_cmd)
        return 0

    def getParameters(self):
        tt.cdm("1D")
        for key, value in self.outdata.items():
            if key not in {'Object Parameter Unit', }:
                cmd = f'o=cds();o.property("{value}")'
                ret = tt.run_js(cmd).get('text')
                self.exportData[key].append(ret)
            with open(self.outPutRetJsonFilePath, 'w') as f:
                json.dump(self.exportData, f, allow_nan=True)

        return 0
    
    def main(self):
        numOfRun = 1000
        num = 0

        keys = self.outdata.keys()
        for idx in keys:
            if idx not in {'Object Parameter Unit', }:
                self.exportData[idx] = []

        for self.i in range(0, len(list(self.indata.values())[0])):
            if num == 0:
                self.startTurboTidesWithApi()
                tt.load(str(Path(self.file).absolute()), True)
                num += 1
            elif num > numOfRun:
                tt.exit()
                num = 1
                # restart and load file
                self.startTurboTidesWithApi()
                tt.load(str(Path(self.file).absolute()), True)

            self.setParameters()
            self.getParameters()

        df = pd.DataFrame.from_dict(self.exportData)
        df.to_pickle('pickle_test.pkl')

        with open(self.outPutRetJsonFilePath, 'w') as f:
            json.dump(self.exportData, f, allow_nan=True)

        tt.exit()
        return 0


def main():
    getData = GetDataFromTurboTides().main()


if __name__ == '__main__':
    main()
