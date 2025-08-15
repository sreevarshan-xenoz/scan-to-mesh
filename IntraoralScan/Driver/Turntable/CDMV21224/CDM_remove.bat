@echo off
set basepath=%CD%
@DPInst64.exe /U %basepath%\ftdibus.inf /S
@DPInst64.exe /U %basepath%\ftdiport.inf /S



